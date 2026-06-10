// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/dab_detr.hpp"

#include <torch/torch.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "detr/models/cond_decoder.hpp"
#include "detr/models/model.hpp"
#include "detr/models/pos_embed.hpp"
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

// Sine embedding of a 4D anchor [..., 4] -> [..., 2d] (concat y, x, w, h parts).
torch::Tensor SineEmbed4D(torch::Tensor pos, std::int64_t d) {
  const double scale = 2.0 * std::numbers::pi;
  const std::int64_t half = d / 2;
  auto dim_t = torch::arange(half, pos.options());
  dim_t = torch::pow(10000.0, 2 * torch::floor(dim_t / 2) / static_cast<double>(half));
  auto embed = [&](torch::Tensor coord) {  // [...] -> [..., half]
    auto p = coord.unsqueeze(-1) * scale / dim_t;
    return torch::stack({p.slice(-1, 0, half, 2).sin(), p.slice(-1, 1, half, 2).cos()}, -1)
        .flatten(-2);
  };
  return torch::cat({embed(pos.select(-1, 1)), embed(pos.select(-1, 0)), embed(pos.select(-1, 2)),
                     embed(pos.select(-1, 3))},
                    -1);  // [..., 2d]
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

class DabDetrImpl : public IModel {
 public:
  explicit DabDetrImpl(Config cfg) : cfg_(cfg) {
    const int d = cfg.hidden_dim;
    backbone_ = register_module(
        "backbone", ResNet(std::vector<int>{3, 4, 6, 3}, /*bottleneck=*/true, /*dc5=*/false));
    input_proj_ = register_module("input_proj", nn::Conv2d(nn::Conv2dOptions(2048, d, 1)));
    refpoint_embed_ = register_module("refpoint_embed", nn::Embedding(cfg.num_queries, 4));
    ref_point_head_ = register_module("ref_point_head", Mlp(2 * d, d, d, 2));
    query_scale_ = register_module("query_scale", Mlp(d, d, d, 2));
    ref_anchor_head_ = register_module("ref_anchor_head", Mlp(d, d, 2, 2));

    encoder_ = register_module("encoder", nn::ModuleList());
    for (int i = 0; i < cfg.enc_layers; ++i) {
      encoder_->push_back(EncoderLayer(d, cfg.nheads, cfg.dim_feedforward));
    }
    decoder_ = register_module("decoder", nn::ModuleList());
    for (int i = 0; i < cfg.dec_layers; ++i) {
      decoder_->push_back(CondDecoderLayer(d, cfg.nheads, cfg.dim_feedforward));
    }
    class_embed_ = register_module("class_embed", nn::Linear(d, cfg.num_classes));
    bbox_embed_ = register_module("bbox_embed", Mlp(d, d, 4, 3));
  }

  Detections Forward(torch::Tensor images) override {
    const int d = cfg_.hidden_dim;
    const std::int64_t half = d / 2;
    auto feat = backbone_->forward(images);
    auto src = input_proj_->forward(feat);
    const auto b = src.size(0);
    const auto h = src.size(2);
    const auto w = src.size(3);

    auto pos = SinePos(b, d, h, w, src.options()).flatten(2).permute({2, 0, 1}).contiguous();
    auto memory = src.flatten(2).permute({2, 0, 1}).contiguous();
    for (const auto& m : *encoder_) {
      memory = m->as<EncoderLayerImpl>()->forward(memory, pos);
    }

    auto anchor = refpoint_embed_->weight.unsqueeze(1).repeat({1, b, 1});  // [nq, B, 4]
    auto reference = anchor.sigmoid();
    auto tgt = torch::zeros({reference.size(0), b, d}, src.options());

    torch::Tensor boxes;
    int layer_id = 0;
    const bool collect_aux = class_embed_->is_training();
    std::vector<torch::Tensor> tgt_layers;
    std::vector<torch::Tensor> box_layers;
    for (const auto& m : *decoder_) {
      auto obj = reference;                              // [nq, B, 4]
      auto sine4 = SineEmbed4D(obj, d);                  // [nq, B, 2d]
      auto query_pos = ref_point_head_->forward(sine4);  // [nq, B, d]
      auto query_sine = sine4.narrow(-1, 0, d);          // [nq, B, d] (y,x parts)
      if (layer_id > 0) {
        query_sine = query_sine * query_scale_->forward(tgt);
      }
      // width/height-modulated positional attention.
      auto ref_hw = ref_anchor_head_->forward(tgt).sigmoid();  // [nq, B, 2]
      auto h_mod = (ref_hw.select(-1, 1) / obj.select(-1, 3)).unsqueeze(-1);
      auto w_mod = (ref_hw.select(-1, 0) / obj.select(-1, 2)).unsqueeze(-1);
      query_sine = torch::cat(
          {query_sine.narrow(-1, 0, half) * h_mod, query_sine.narrow(-1, half, half) * w_mod}, -1);

      tgt = m->as<CondDecoderLayerImpl>()->forward(tgt, memory, pos, query_pos, query_sine,
                                                   layer_id == 0);
      // iterative anchor refinement.
      auto new_ref = (bbox_embed_->forward(tgt) + InverseSigmoid(reference)).sigmoid();
      boxes = new_ref;
      if (collect_aux) {
        tgt_layers.push_back(tgt);
        box_layers.push_back(new_ref);
      }
      reference = new_ref.detach();
      ++layer_id;
    }

    Detections det;
    det.logits = class_embed_->forward(tgt.transpose(0, 1));  // [B, nq, num_classes]
    det.boxes = boxes.transpose(0, 1);                        // [B, nq, 4] cxcywh
    // Deep supervision: per-layer class head + the layer's refined anchor box.
    for (std::size_t i = 0; collect_aux && i + 1 < tgt_layers.size(); ++i) {
      det.aux_logits.push_back(class_embed_->forward(tgt_layers[i].transpose(0, 1)));
      det.aux_boxes.push_back(box_layers[i].transpose(0, 1));
    }
    return det;
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = "dab-detr";
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.focal = true;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/IDEA-Research/DAB-DETR";
    return m;
  }

 private:
  Config cfg_;
  ResNet backbone_{nullptr};
  nn::Conv2d input_proj_{nullptr};
  nn::Embedding refpoint_embed_{nullptr};
  Mlp ref_point_head_{nullptr};
  Mlp query_scale_{nullptr};
  Mlp ref_anchor_head_{nullptr};
  nn::ModuleList encoder_{nullptr};
  nn::ModuleList decoder_{nullptr};
  nn::Linear class_embed_{nullptr};
  Mlp bbox_embed_{nullptr};
};

}  // namespace

std::shared_ptr<IModel> MakeDabDetr(const YAML::Node& cfg) {
  return std::make_shared<DabDetrImpl>(ReadConfig(cfg));
}

ModelMeta DabDetrMeta(const YAML::Node& cfg) {
  Config c = ReadConfig(cfg);
  ModelMeta m;
  m.name = "dab-detr";
  m.imgsz = c.imgsz;
  m.num_classes = c.num_classes;
  m.num_queries = c.num_queries;
  m.focal = true;
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/IDEA-Research/DAB-DETR";
  return m;
}

}  // namespace detr::models
