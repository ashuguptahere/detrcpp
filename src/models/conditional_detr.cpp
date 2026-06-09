// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/conditional_detr.hpp"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "detr/models/cond_decoder.hpp"
#include "detr/models/model.hpp"
#include "detr/models/registry.hpp"
#include "detr/models/resnet.hpp"

namespace detr::models {

namespace {

namespace nn = torch::nn;

struct Config {
  int hidden_dim{256};
  int nheads{8};
  int enc_layers{6};
  int dec_layers{6};
  int dim_feedforward{2048};
  int num_queries{300};
  int num_classes{91};
  int imgsz{640};
  std::vector<int> backbone_blocks{3, 4, 6, 3};
};

template <typename T>
T Get(const YAML::Node& c, const char* k, T fb) {
  return (c && c[k]) ? c[k].as<T>() : fb;
}

Config ReadConfig(const YAML::Node& c) {
  Config x;
  x.hidden_dim = Get(c, "hidden_dim", x.hidden_dim);
  x.nheads = Get(c, "nheads", x.nheads);
  x.enc_layers = Get(c, "enc_layers", x.enc_layers);
  x.dec_layers = Get(c, "dec_layers", x.dec_layers);
  x.dim_feedforward = Get(c, "dim_feedforward", x.dim_feedforward);
  x.num_queries = Get(c, "num_queries", x.num_queries);
  x.num_classes = Get(c, "num_classes", x.num_classes);
  x.imgsz = Get(c, "imgsz", x.imgsz);
  return x;
}

torch::Tensor InverseSigmoid(torch::Tensor x, double eps = 1e-5) {
  x = x.clamp(0, 1);
  return torch::log(x.clamp_min(eps) / (1 - x).clamp_min(eps));
}

// 2D image sine positional embedding (DETR), no mask: [B, d, H, W].
torch::Tensor SinePos(std::int64_t b, std::int64_t d, std::int64_t h, std::int64_t w,
                      const torch::TensorOptions& opts) {
  constexpr double kPi = 3.14159265358979323846;
  const std::int64_t half = d / 2;
  const double scale = 2.0 * kPi;
  auto ys = torch::arange(1, h + 1, opts) / (static_cast<double>(h) + 1e-6) * scale;
  auto xs = torch::arange(1, w + 1, opts) / (static_cast<double>(w) + 1e-6) * scale;
  auto dim_t = torch::arange(0, half, opts);
  dim_t = torch::pow(10000.0, 2 * torch::floor(dim_t / 2) / static_cast<double>(half));
  auto px = xs.unsqueeze(1) / dim_t.unsqueeze(0);
  auto py = ys.unsqueeze(1) / dim_t.unsqueeze(0);
  auto interleave = [half](torch::Tensor p) {
    return torch::stack({p.slice(1, 0, half, 2).sin(), p.slice(1, 1, half, 2).cos()}, 2).flatten(1);
  };
  px = interleave(px);
  py = interleave(py);
  auto pyf = py.unsqueeze(1).expand({h, w, half});
  auto pxf = px.unsqueeze(0).expand({h, w, half});
  auto pos = torch::cat({pyf, pxf}, 2);
  return pos.permute({2, 0, 1}).unsqueeze(0).expand({b, d, h, w}).contiguous();
}

// Sine embedding of a 2D reference point: [..., 2] -> [..., d].
torch::Tensor SineEmbedForRef(torch::Tensor pos, std::int64_t d) {
  constexpr double kPi = 3.14159265358979323846;
  const double scale = 2.0 * kPi;
  const std::int64_t half = d / 2;
  auto dim_t = torch::arange(half, pos.options());
  dim_t = torch::pow(10000.0, 2 * torch::floor(dim_t / 2) / static_cast<double>(half));
  auto x = pos.select(-1, 0).unsqueeze(-1) * scale / dim_t;  // [..., half]
  auto y = pos.select(-1, 1).unsqueeze(-1) * scale / dim_t;
  auto interleave = [half](torch::Tensor p) {
    return torch::stack({p.slice(-1, 0, half, 2).sin(), p.slice(-1, 1, half, 2).cos()}, -1)
        .flatten(-2);
  };
  return torch::cat({interleave(y), interleave(x)}, -1);  // [..., d]
}

struct MlpImpl : nn::Module {
  nn::ModuleList layers{nullptr};
  int n_;
  MlpImpl(int in, int hidden, int out, int n) : n_(n) {
    layers = register_module("layers", nn::ModuleList());
    int prev = in;
    for (int i = 0; i < n; ++i) {
      layers->push_back(nn::Linear(prev, (i + 1 == n) ? out : hidden));
      prev = (i + 1 == n) ? out : hidden;
    }
  }
  torch::Tensor forward(torch::Tensor x) {
    for (int i = 0; i < n_; ++i) {
      x = layers[static_cast<std::size_t>(i)]->as<nn::LinearImpl>()->forward(x);
      if (i + 1 < n_) {
        x = torch::relu(x);
      }
    }
    return x;
  }
};
TORCH_MODULE(Mlp);

struct EncoderLayerImpl : nn::Module {
  nn::MultiheadAttention self_attn{nullptr};
  nn::Linear linear1{nullptr};
  nn::Linear linear2{nullptr};
  nn::LayerNorm norm1{nullptr};
  nn::LayerNorm norm2{nullptr};
  EncoderLayerImpl(int d, int nhead, int ff) {
    self_attn = register_module(
        "self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, nhead).dropout(0.1)));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
  }
  torch::Tensor forward(torch::Tensor src, const torch::Tensor& pos) {
    auto q = src + pos;
    auto attn = std::get<0>(self_attn->forward(q, q, src));
    src = norm1->forward(src + attn);
    auto ff = linear2->forward(torch::relu(linear1->forward(src)));
    return norm2->forward(src + ff);
  }
};
TORCH_MODULE(EncoderLayer);

class ConditionalDetrImpl : public IModel {
 public:
  explicit ConditionalDetrImpl(Config cfg) : cfg_(cfg) {
    const int d = cfg.hidden_dim;
    backbone_ = register_module("backbone",
                                ResNet(cfg.backbone_blocks, /*bottleneck=*/true, /*dc5=*/false));
    input_proj_ = register_module("input_proj", nn::Conv2d(nn::Conv2dOptions(2048, d, 1)));
    query_embed_ = register_module("query_embed", nn::Embedding(cfg.num_queries, d));
    ref_point_head_ = register_module("ref_point_head", Mlp(d, d, 2, 2));
    query_scale_ = register_module("query_scale", Mlp(d, d, d, 2));

    encoder_ = register_module("encoder", nn::ModuleList());
    for (int i = 0; i < cfg.enc_layers; ++i) {
      encoder_->push_back(EncoderLayer(d, cfg.nheads, cfg.dim_feedforward));
    }
    decoder_ = register_module("decoder", nn::ModuleList());
    for (int i = 0; i < cfg.dec_layers; ++i) {
      decoder_->push_back(CondDecoderLayer(d, cfg.nheads, cfg.dim_feedforward));
    }
    decoder_norm_ = register_module("decoder_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
    class_embed_ = register_module("class_embed", nn::Linear(d, cfg.num_classes));
    bbox_embed_ = register_module("bbox_embed", Mlp(d, d, 4, 3));
  }

  Detections Forward(torch::Tensor images) override {
    const int d = cfg_.hidden_dim;
    auto feat = backbone_->forward(images);
    auto src = input_proj_->forward(feat);
    const auto b = src.size(0);
    const auto h = src.size(2);
    const auto w = src.size(3);

    auto pos = SinePos(b, d, h, w, src.options()).flatten(2).permute({2, 0, 1}).contiguous();
    auto memory = src.flatten(2).permute({2, 0, 1}).contiguous();  // [hw, B, d]
    for (const auto& m : *encoder_) {
      memory = m->as<EncoderLayerImpl>()->forward(memory, pos);
    }

    auto query_pos = query_embed_->weight.unsqueeze(1).repeat({1, b, 1});  // [nq, B, d]
    auto tgt = torch::zeros_like(query_pos);
    auto reference = ref_point_head_->forward(query_pos).sigmoid();  // [nq, B, 2]

    int layer_id = 0;
    for (const auto& m : *decoder_) {
      auto query_sine = SineEmbedForRef(reference, d);  // [nq, B, d]
      if (layer_id > 0) {
        query_sine = query_sine * query_scale_->forward(tgt);
      }
      tgt = m->as<CondDecoderLayerImpl>()->forward(tgt, memory, pos, query_pos, query_sine,
                                                   layer_id == 0);
      ++layer_id;
    }

    tgt = decoder_norm_->forward(tgt);                            // final decoder LayerNorm
    auto hs = tgt.transpose(0, 1);                                // [B, nq, d]
    auto ref_before = InverseSigmoid(reference).transpose(0, 1);  // [B, nq, 2]
    auto box = bbox_embed_->forward(hs);                          // [B, nq, 4]
    box = box + torch::cat({ref_before, torch::zeros_like(ref_before)}, -1);

    Detections det;
    det.logits = class_embed_->forward(hs);  // [B, nq, num_classes] (sigmoid/focal)
    det.boxes = box.sigmoid();
    return det;
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = "conditional-detr";
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.focal = true;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/Atten4Vis/ConditionalDETR";
    return m;
  }

 private:
  Config cfg_;
  ResNet backbone_{nullptr};
  nn::Conv2d input_proj_{nullptr};
  nn::Embedding query_embed_{nullptr};
  Mlp ref_point_head_{nullptr};
  Mlp query_scale_{nullptr};
  nn::LayerNorm decoder_norm_{nullptr};
  nn::ModuleList encoder_{nullptr};
  nn::ModuleList decoder_{nullptr};
  nn::Linear class_embed_{nullptr};
  Mlp bbox_embed_{nullptr};
};

}  // namespace

std::shared_ptr<IModel> MakeConditionalDetr(const YAML::Node& cfg) {
  return std::make_shared<ConditionalDetrImpl>(ReadConfig(cfg));
}

ModelMeta ConditionalDetrMeta(const YAML::Node& cfg) {
  Config c = ReadConfig(cfg);
  ModelMeta m;
  m.name = "conditional-detr";
  m.imgsz = c.imgsz;
  m.num_classes = c.num_classes;
  m.num_queries = c.num_queries;
  m.focal = true;
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/Atten4Vis/ConditionalDETR";
  return m;
}

}  // namespace detr::models
