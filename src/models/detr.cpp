// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/detr.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "detr/models/model.hpp"
#include "detr/models/registry.hpp"

namespace detr::models {

namespace {

namespace nn = torch::nn;

struct Config {
  int hidden_dim{256};
  int nheads{8};
  int enc_layers{6};
  int dec_layers{6};
  int dim_feedforward{2048};
  int num_queries{100};
  int num_classes{91};
  int imgsz{640};
  int backbone_width{64};
};

template <typename T>
T Get(const YAML::Node& cfg, const char* key, T fallback) {
  if (cfg && cfg[key]) {
    return cfg[key].as<T>();
  }
  return fallback;
}

Config ReadConfig(const YAML::Node& cfg) {
  Config c;
  c.hidden_dim = Get(cfg, "hidden_dim", c.hidden_dim);
  c.nheads = Get(cfg, "nheads", c.nheads);
  c.enc_layers = Get(cfg, "enc_layers", c.enc_layers);
  c.dec_layers = Get(cfg, "dec_layers", c.dec_layers);
  c.dim_feedforward = Get(cfg, "dim_feedforward", c.dim_feedforward);
  c.num_queries = Get(cfg, "num_queries", c.num_queries);
  c.num_classes = Get(cfg, "num_classes", c.num_classes);
  c.imgsz = Get(cfg, "imgsz", c.imgsz);
  c.backbone_width = Get(cfg, "backbone_width", c.backbone_width);
  return c;
}

// 2D sine positional embedding (DETR's PositionEmbeddingSine, no padding mask).
// Returns [B, d, h, w].
torch::Tensor SinePos(std::int64_t b, std::int64_t d, std::int64_t h, std::int64_t w,
                      const torch::TensorOptions& opts) {
  constexpr double kPi = 3.14159265358979323846;
  const std::int64_t half = d / 2;
  const double scale = 2.0 * kPi;
  auto ys = torch::arange(1, h + 1, opts) / (static_cast<double>(h) + 1e-6) * scale;  // [h]
  auto xs = torch::arange(1, w + 1, opts) / (static_cast<double>(w) + 1e-6) * scale;  // [w]
  auto dim_t = torch::arange(0, half, opts);
  dim_t = torch::pow(10000.0, 2 * torch::floor(dim_t / 2) / static_cast<double>(half));  // [half]

  auto px = xs.unsqueeze(1) / dim_t.unsqueeze(0);  // [w, half]
  auto py = ys.unsqueeze(1) / dim_t.unsqueeze(0);  // [h, half]
  auto interleave = [half](torch::Tensor p) {
    return torch::stack({p.slice(1, 0, half, 2).sin(), p.slice(1, 1, half, 2).cos()}, 2)
        .flatten(1);  // [n, half]
  };
  px = interleave(px);  // [w, half]
  py = interleave(py);  // [h, half]
  auto pyf = py.unsqueeze(1).expand({h, w, half});  // [h, w, half]
  auto pxf = px.unsqueeze(0).expand({h, w, half});  // [h, w, half]
  auto pos = torch::cat({pyf, pxf}, 2);             // [h, w, d]
  return pos.permute({2, 0, 1}).unsqueeze(0).expand({b, d, h, w}).contiguous();
}

nn::Sequential MakeBackbone(int w0) {
  // A compact but real conv backbone downsampling the input by 32x. Channels
  // grow w0 -> 8*w0 across four stages after a ResNet-style stem.
  auto conv = [](int in, int out, int k, int s, int p) {
    return nn::Conv2d(nn::Conv2dOptions(in, out, k).stride(s).padding(p).bias(false));
  };
  nn::Sequential s;
  s->push_back(conv(3, w0, 7, 2, 3));
  s->push_back(nn::BatchNorm2d(w0));
  s->push_back(nn::Functional(torch::relu));
  s->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2).padding(1)));
  const int chans[4] = {w0, 2 * w0, 4 * w0, 8 * w0};
  int in = w0;
  for (int i = 0; i < 4; ++i) {
    const int out = chans[i];
    const int stride = (i == 0) ? 1 : 2;
    s->push_back(conv(in, out, 3, stride, 1));
    s->push_back(nn::BatchNorm2d(out));
    s->push_back(nn::Functional(torch::relu));
    s->push_back(conv(out, out, 3, 1, 1));
    s->push_back(nn::BatchNorm2d(out));
    s->push_back(nn::Functional(torch::relu));
    in = out;
  }
  return s;
}

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
    src = norm2->forward(src + ff);
    return src;
  }
};
TORCH_MODULE(EncoderLayer);

struct DecoderLayerImpl : nn::Module {
  nn::MultiheadAttention self_attn{nullptr};
  nn::MultiheadAttention cross_attn{nullptr};
  nn::Linear linear1{nullptr};
  nn::Linear linear2{nullptr};
  nn::LayerNorm norm1{nullptr};
  nn::LayerNorm norm2{nullptr};
  nn::LayerNorm norm3{nullptr};

  DecoderLayerImpl(int d, int nhead, int ff) {
    self_attn = register_module(
        "self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, nhead).dropout(0.1)));
    cross_attn = register_module(
        "cross_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, nhead).dropout(0.1)));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm3 = register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({d})));
  }

  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& memory,
                        const torch::Tensor& pos, const torch::Tensor& query_pos) {
    auto q = tgt + query_pos;
    auto sa = std::get<0>(self_attn->forward(q, q, tgt));
    tgt = norm1->forward(tgt + sa);
    auto ca = std::get<0>(cross_attn->forward(tgt + query_pos, memory + pos, memory));
    tgt = norm2->forward(tgt + ca);
    auto ff = linear2->forward(torch::relu(linear1->forward(tgt)));
    tgt = norm3->forward(tgt + ff);
    return tgt;
  }
};
TORCH_MODULE(DecoderLayer);

class DetrImpl : public IModel {
 public:
  explicit DetrImpl(Config cfg) : cfg_(cfg) {
    backbone_ = register_module("backbone", MakeBackbone(cfg.backbone_width));
    input_proj_ = register_module(
        "input_proj", nn::Conv2d(nn::Conv2dOptions(8 * cfg.backbone_width, cfg.hidden_dim, 1)));
    query_embed_ = register_module("query_embed", nn::Embedding(cfg.num_queries, cfg.hidden_dim));

    encoder_ = register_module("encoder", nn::ModuleList());
    for (int i = 0; i < cfg.enc_layers; ++i) {
      encoder_->push_back(EncoderLayer(cfg.hidden_dim, cfg.nheads, cfg.dim_feedforward));
    }
    decoder_ = register_module("decoder", nn::ModuleList());
    for (int i = 0; i < cfg.dec_layers; ++i) {
      decoder_->push_back(DecoderLayer(cfg.hidden_dim, cfg.nheads, cfg.dim_feedforward));
    }

    class_embed_ =
        register_module("class_embed", nn::Linear(cfg.hidden_dim, cfg.num_classes + 1));
    bbox_embed_ = register_module(
        "bbox_embed",
        nn::Sequential(nn::Linear(cfg.hidden_dim, cfg.hidden_dim), nn::Functional(torch::relu),
                       nn::Linear(cfg.hidden_dim, cfg.hidden_dim), nn::Functional(torch::relu),
                       nn::Linear(cfg.hidden_dim, 4)));
  }

  Detections Forward(torch::Tensor images) override {
    auto feat = backbone_->forward(images);     // [B, 8w, h, w]
    auto src = input_proj_->forward(feat);       // [B, d, h, w]
    const auto b = src.size(0);
    const auto d = src.size(1);
    const auto h = src.size(2);
    const auto w = src.size(3);

    auto pos = SinePos(b, d, h, w, src.options());   // [B, d, h, w]
    // [B, d, h, w] -> [h*w, B, d]
    auto src_seq = src.flatten(2).permute({2, 0, 1}).contiguous();
    auto pos_seq = pos.flatten(2).permute({2, 0, 1}).contiguous();

    auto memory = src_seq;
    for (const auto& m : *encoder_) {
      memory = m->as<EncoderLayerImpl>()->forward(memory, pos_seq);
    }

    auto query = query_embed_->weight.unsqueeze(1).repeat({1, b, 1});  // [Q, B, d]
    auto tgt = torch::zeros_like(query);
    for (const auto& m : *decoder_) {
      tgt = m->as<DecoderLayerImpl>()->forward(tgt, memory, pos_seq, query);
    }

    auto hs = tgt.transpose(0, 1);                       // [B, Q, d]
    Detections out;
    out.logits = class_embed_->forward(hs);              // [B, Q, C+1]
    out.boxes = bbox_embed_->forward(hs).sigmoid();      // [B, Q, 4]
    return out;
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = "detr";
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/facebookresearch/detr";
    return m;
  }

 private:
  Config cfg_;
  nn::Sequential backbone_{nullptr};
  nn::Conv2d input_proj_{nullptr};
  nn::Embedding query_embed_{nullptr};
  nn::ModuleList encoder_{nullptr};
  nn::ModuleList decoder_{nullptr};
  nn::Linear class_embed_{nullptr};
  nn::Sequential bbox_embed_{nullptr};
};

}  // namespace

std::shared_ptr<IModel> MakeDetr(const YAML::Node& cfg) {
  return std::make_shared<DetrImpl>(ReadConfig(cfg));
}

ModelMeta DetrMeta(const YAML::Node& cfg) {
  Config c = ReadConfig(cfg);
  ModelMeta m;
  m.name = "detr";
  m.imgsz = c.imgsz;
  m.num_classes = c.num_classes;
  m.num_queries = c.num_queries;
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/facebookresearch/detr";
  return m;
}

void RegisterBuiltins() {
  Registry::Instance().Register("detr", DetrMeta({}), &MakeDetr);
}

}  // namespace detr::models
