// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/conditional_detr.hpp"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

// Multi-head attention with externally-projected q/k/v (no in_proj), allowing
// q/k of one width and v of another (the decoupled cross-attention). Inputs are
// [L, B, dim]; returns [Lq, B, v_dim].
torch::Tensor MultiHeadAttn(torch::Tensor q, torch::Tensor k, torch::Tensor v, int nhead) {
  const auto lq = q.size(0);
  const auto b = q.size(1);
  const auto lk = k.size(0);
  const auto qk_hd = q.size(2) / nhead;
  const auto v_hd = v.size(2) / nhead;
  auto split = [&](torch::Tensor t, std::int64_t l, std::int64_t hd) {
    return t.view({l, b, nhead, hd}).permute({1, 2, 0, 3}).reshape({b * nhead, l, hd});
  };
  auto qh = split(q, lq, qk_hd);
  auto kh = split(k, lk, qk_hd);
  auto vh = split(v, lk, v_hd);
  auto scores = torch::bmm(qh, kh.transpose(1, 2)) / std::sqrt(static_cast<double>(qk_hd));
  auto out = torch::bmm(scores.softmax(-1), vh);  // [b*nh, lq, v_hd]
  return out.view({b, nhead, lq, v_hd}).permute({2, 0, 1, 3}).reshape({lq, b, nhead * v_hd});
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

// Conditional decoder layer: standard self-attn (externally projected) + the
// decoupled content/spatial cross-attention.
struct DecoderLayerImpl : nn::Module {
  nn::Linear sa_qcontent{nullptr}, sa_qpos{nullptr}, sa_kcontent{nullptr}, sa_kpos{nullptr},
      sa_v{nullptr}, sa_out{nullptr};
  nn::Linear ca_qcontent{nullptr}, ca_qpos{nullptr}, ca_kcontent{nullptr}, ca_kpos{nullptr},
      ca_v{nullptr}, ca_qpos_sine{nullptr}, ca_out{nullptr};
  nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
  nn::Linear linear1{nullptr}, linear2{nullptr};
  int nhead_;
  int d_;

  DecoderLayerImpl(int d, int nhead, int ff) : nhead_(nhead), d_(d) {
    auto lin = [&](const char* n) { return register_module(n, nn::Linear(d, d)); };
    sa_qcontent = lin("sa_qcontent_proj");
    sa_qpos = lin("sa_qpos_proj");
    sa_kcontent = lin("sa_kcontent_proj");
    sa_kpos = lin("sa_kpos_proj");
    sa_v = lin("sa_v_proj");
    sa_out = lin("sa_out_proj");
    ca_qcontent = lin("ca_qcontent_proj");
    ca_qpos = lin("ca_qpos_proj");
    ca_kcontent = lin("ca_kcontent_proj");
    ca_kpos = lin("ca_kpos_proj");
    ca_v = lin("ca_v_proj");
    ca_qpos_sine = lin("ca_qpos_sine_proj");
    ca_out = lin("ca_out_proj");
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm3 = register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({d})));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
  }

  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& memory, const torch::Tensor& pos,
                        const torch::Tensor& query_pos, const torch::Tensor& query_sine,
                        bool is_first) {
    const auto nq = tgt.size(0);
    const auto b = tgt.size(1);
    const auto hw = memory.size(0);
    const auto hd = d_ / nhead_;

    // self-attention (q/k/v projected externally; no in_proj).
    auto q = sa_qcontent->forward(tgt) + sa_qpos->forward(query_pos);
    auto k = sa_kcontent->forward(tgt) + sa_kpos->forward(query_pos);
    auto sa = sa_out->forward(MultiHeadAttn(q, k, sa_v->forward(tgt), nhead_));
    tgt = norm1->forward(tgt + sa);

    // conditional cross-attention.
    auto qc = ca_qcontent->forward(tgt);
    auto kc = ca_kcontent->forward(memory);
    auto v = ca_v->forward(memory);
    auto kp = ca_kpos->forward(pos);
    torch::Tensor qq;
    torch::Tensor kk;
    if (is_first) {
      qq = qc + ca_qpos->forward(query_pos);
      kk = kc + kp;
    } else {
      qq = qc;
      kk = kc;
    }
    auto qse = ca_qpos_sine->forward(query_sine).view({nq, b, nhead_, hd});
    auto qcat = torch::cat({qq.view({nq, b, nhead_, hd}), qse}, 3).view({nq, b, 2 * d_});
    auto kcat = torch::cat({kk.view({hw, b, nhead_, hd}), kp.view({hw, b, nhead_, hd})}, 3)
                    .view({hw, b, 2 * d_});
    auto ca = ca_out->forward(MultiHeadAttn(qcat, kcat, v, nhead_));
    tgt = norm2->forward(tgt + ca);

    auto ff = linear2->forward(torch::relu(linear1->forward(tgt)));
    return norm3->forward(tgt + ff);
  }
};
TORCH_MODULE(DecoderLayer);

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
      decoder_->push_back(DecoderLayer(d, cfg.nheads, cfg.dim_feedforward));
    }
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
      tgt = m->as<DecoderLayerImpl>()->forward(tgt, memory, pos, query_pos, query_sine,
                                               layer_id == 0);
      ++layer_id;
    }

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
