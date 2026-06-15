// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/anchor_detr.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "detr/models/registry.hpp"

namespace detr::models {

namespace {

namespace I = torch::indexing;

// Sinusoidal embedding of a 1D coordinate in [0,1] -> [..., num_feats]. Matches the
// upstream pos2posemb1d (interleaved sin(even)/cos(odd)).
torch::Tensor Pos2Posemb1d(const torch::Tensor& pos, int num_feats = 256) {
  const double scale = 2.0 * M_PI;
  auto p = pos * scale;
  auto k = torch::floor(torch::arange(num_feats, pos.options()) / 2);
  auto dim_t = torch::pow(10000.0, 2.0 * k / num_feats);  // [num_feats]
  auto px = p.unsqueeze(-1) / dim_t;                       // [..., num_feats]
  auto even = px.slice(-1, 0, num_feats, 2).sin();
  auto odd = px.slice(-1, 1, num_feats, 2).cos();
  return torch::stack({even, odd}, -1).flatten(-2);
}

// Sinusoidal embedding of a 2D point in [0,1]^2 -> [..., 2*num_feats] (concat y then x).
torch::Tensor Pos2Posemb2d(const torch::Tensor& pos, int num_feats = 128) {
  const double scale = 2.0 * M_PI;
  auto p = pos * scale;
  auto k = torch::floor(torch::arange(num_feats, pos.options()) / 2);
  auto dim_t = torch::pow(10000.0, 2.0 * k / num_feats);
  auto enc = [&](const torch::Tensor& c) {
    auto px = c.unsqueeze(-1) / dim_t;
    return torch::stack({px.slice(-1, 0, num_feats, 2).sin(), px.slice(-1, 1, num_feats, 2).cos()}, -1)
        .flatten(-2);
  };
  auto pos_x = enc(p.index({"...", 0}));
  auto pos_y = enc(p.index({"...", 1}));
  return torch::cat({pos_y, pos_x}, -1);
}

torch::Tensor InverseSigmoid(torch::Tensor x, double eps = 1e-5) {
  x = x.clamp(0.0, 1.0);
  return torch::log(x.clamp_min(eps) / (1.0 - x).clamp_min(eps));
}

}  // namespace

RcdaImpl::RcdaImpl(int d_model, int nhead) : nhead_(nhead), d_model_(d_model) {
  in_proj_weight = register_parameter("in_proj_weight", torch::empty({5 * d_model, d_model}));
  in_proj_bias = register_parameter("in_proj_bias", torch::zeros({5 * d_model}));
  out_proj = register_module("out_proj", nn::Linear(d_model, d_model));
}

torch::Tensor RcdaImpl::forward(const torch::Tensor& query_row, const torch::Tensor& query_col,
                                const torch::Tensor& key_row, const torch::Tensor& key_col,
                                const torch::Tensor& value) {
  const int E = d_model_, nh = nhead_, hd = E / nh;
  const auto B = query_row.size(0), L = query_row.size(1);
  const auto H = key_row.size(1), W = key_row.size(2);
  auto w = [&](int i) { return in_proj_weight.slice(0, i * E, (i + 1) * E); };
  auto b = [&](int i) { return in_proj_bias.slice(0, i * E, (i + 1) * E); };
  auto qr = (torch::linear(query_row, w(0), b(0))).reshape({B, L, nh, hd});
  auto qc = (torch::linear(query_col, w(1), b(1))).reshape({B, L, nh, hd});
  auto kr = torch::linear(key_row, w(2), b(2)).mean(1).reshape({B, W, nh, hd});  // mean over H
  auto kc = torch::linear(key_col, w(3), b(3)).mean(2).reshape({B, H, nh, hd});  // mean over W
  auto v = torch::linear(value, w(4), b(4)).reshape({B, H, W, nh, hd});
  const double scaling = std::pow(static_cast<double>(hd), -0.5);
  auto attn_row = torch::softmax(torch::einsum("blnd,bwnd->bnlw", {qr * scaling, kr}), -1);
  auto attn_col = torch::softmax(torch::einsum("blnd,bond->bnlo", {qc * scaling, kc}), -1);
  // out[b,l,n,d] = sum_{o(H),w(W)} attn_col[b,n,l,o] * attn_row[b,n,l,w] * v[b,o,w,n,d]
  auto out = torch::einsum("bnlo,bnlw,bownd->blnd", {attn_col, attn_row, v}).reshape({B, L, E});
  return out_proj->forward(out);
}

AdaptPosImpl::AdaptPosImpl(int d_model) {
  fc0 = register_module("0", nn::Linear(d_model, d_model));
  fc2 = register_module("2", nn::Linear(d_model, d_model));
}
torch::Tensor AdaptPosImpl::forward(const torch::Tensor& x) {
  return fc2->forward(torch::relu(fc0->forward(x)));
}

AnchorFfnImpl::AnchorFfnImpl(int d_model, int d_ffn) {
  linear1 = register_module("linear1", nn::Linear(d_model, d_ffn));
  linear2 = register_module("linear2", nn::Linear(d_ffn, d_model));
  norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d_model})));
}
torch::Tensor AnchorFfnImpl::forward(torch::Tensor x) {
  auto x2 = linear2->forward(torch::relu(linear1->forward(x)));
  return norm2->forward(x + x2);
}

AnchorEncoderLayerImpl::AnchorEncoderLayerImpl(int d_model, int d_ffn, int nhead) {
  self_attn = register_module("self_attn", Rcda(d_model, nhead));
  norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d_model})));
  ffn = register_module("ffn", AnchorFfn(d_model, d_ffn));
}
torch::Tensor AnchorEncoderLayerImpl::forward(torch::Tensor src, const torch::Tensor& posemb_row,
                                              const torch::Tensor& posemb_col) {
  const auto B = src.size(0), C = src.size(1), H = src.size(2), W = src.size(3);
  src = src.permute({0, 2, 3, 1});  // [B, H, W, C]
  auto pr = posemb_row.unsqueeze(1).expand({B, H, W, C});
  auto pc = posemb_col.unsqueeze(2).expand({B, H, W, C});
  auto q_row = (src + pr).reshape({B, H * W, C});
  auto q_col = (src + pc).reshape({B, H * W, C});
  auto src2 = self_attn->forward(q_row, q_col, src + pr, src + pc, src).reshape({B, H, W, C});
  src = norm1->forward(src + src2);
  src = ffn->forward(src);
  return src.permute({0, 3, 1, 2});  // [B, C, H, W]
}

AnchorDecoderLayerImpl::AnchorDecoderLayerImpl(int d_model, int d_ffn, int nhead) : nhead_(nhead) {
  cross_attn = register_module("cross_attn", Rcda(d_model, nhead));
  norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d_model})));
  self_attn = register_module(
      "self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d_model, nhead)));
  norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d_model})));
  ffn = register_module("ffn", AnchorFfn(d_model, d_ffn));
}
torch::Tensor AnchorDecoderLayerImpl::forward(torch::Tensor tgt,
                                              const torch::Tensor& reference_points,
                                              const torch::Tensor& srcs,
                                              const torch::Tensor& posemb_row,
                                              const torch::Tensor& posemb_col, AdaptPos adapt_pos2d,
                                              AdaptPos adapt_pos1d) {
  auto query_pos = adapt_pos2d->forward(Pos2Posemb2d(reference_points));  // [B, nq, C]
  // self-attention (LibTorch MultiheadAttention is sequence-first [L, N, E]).
  auto q = (tgt + query_pos).transpose(0, 1);
  auto tgt2 = std::get<0>(self_attn->forward(q, q, tgt.transpose(0, 1))).transpose(0, 1);
  tgt = norm2->forward(tgt + tgt2);
  // RCDA cross-attention into the feature map.
  const auto B = srcs.size(0), C = srcs.size(1), H = srcs.size(2), W = srcs.size(3);
  auto sp = srcs.permute({0, 2, 3, 1});  // [B, H, W, C]
  auto qpx = adapt_pos1d->forward(Pos2Posemb1d(reference_points.index({"...", 0})));
  auto qpy = adapt_pos1d->forward(Pos2Posemb1d(reference_points.index({"...", 1})));
  auto pr = posemb_row.unsqueeze(1).expand({B, H, W, C});
  auto pc = posemb_col.unsqueeze(2).expand({B, H, W, C});
  auto tgt2c = cross_attn->forward(tgt + qpx, tgt + qpy, sp + pr, sp + pc, sp);
  tgt = norm1->forward(tgt + tgt2c);
  return ffn->forward(tgt);
}

AnchorTransformerImpl::AnchorTransformerImpl(const AnchorTransformerConfig& cfg) : cfg_(cfg) {
  const int d = cfg.d_model;
  pattern = register_module("pattern", nn::Embedding(cfg.num_pattern, d));
  position = register_module("position", nn::Embedding(cfg.num_position, 2));
  adapt_pos1d = register_module("adapt_pos1d", AdaptPos(d));
  adapt_pos2d = register_module("adapt_pos2d", AdaptPos(d));
  encoder_layers = register_module("encoder_layers", nn::ModuleList());
  for (int i = 0; i < cfg.enc_layers; ++i)
    encoder_layers->push_back(AnchorEncoderLayer(d, cfg.dim_feedforward, cfg.nhead));
  decoder_layers = register_module("decoder_layers", nn::ModuleList());
  for (int i = 0; i < cfg.dec_layers; ++i)
    decoder_layers->push_back(AnchorDecoderLayer(d, cfg.dim_feedforward, cfg.nhead));
  class_embed = register_module("class_embed", nn::Linear(d, cfg.num_classes));
  auto bbox_holder = register_module("bbox_embed", std::make_shared<nn::Module>());
  bbox_embed = bbox_holder->register_module("layers", nn::ModuleList());
  bbox_embed->push_back(nn::Linear(d, d));
  bbox_embed->push_back(nn::Linear(d, d));
  bbox_embed->push_back(nn::Linear(d, 4));
}

std::pair<torch::Tensor, torch::Tensor> AnchorTransformerImpl::forward(const torch::Tensor& src) {
  const auto B = src.size(0);
  const int C = cfg_.d_model;
  const auto H = src.size(2), W = src.size(3);
  // Anchor points (learned) instantiated with each content pattern -> queries.
  auto reference_points = position->weight.unsqueeze(0).repeat({B, cfg_.num_pattern, 1});  // [B,nq,2]
  auto tgt = pattern->weight.reshape({1, cfg_.num_pattern, 1, C})
                 .repeat({B, 1, cfg_.num_position, 1})
                 .reshape({B, cfg_.num_pattern * cfg_.num_position, C});
  // Row/column position embeddings (full-valid feature map, no padding).
  auto opt = src.options();
  auto pos_row = ((torch::arange(W, opt) + 0.5) / W).unsqueeze(0).expand({B, W});
  auto pos_col = ((torch::arange(H, opt) + 0.5) / H).unsqueeze(0).expand({B, H});
  auto posemb_row = adapt_pos1d->forward(Pos2Posemb1d(pos_row));  // [B, W, C]
  auto posemb_col = adapt_pos1d->forward(Pos2Posemb1d(pos_col));  // [B, H, C]

  auto outputs = src;
  for (std::size_t i = 0; i < encoder_layers->size(); ++i)
    outputs = encoder_layers[i]->as<AnchorEncoderLayerImpl>()->forward(outputs, posemb_row, posemb_col);

  auto output = tgt;
  torch::Tensor out_logits, out_boxes;
  for (std::size_t i = 0; i < decoder_layers->size(); ++i) {
    output = decoder_layers[i]->as<AnchorDecoderLayerImpl>()->forward(
        output, reference_points, outputs, posemb_row, posemb_col, adapt_pos2d, adapt_pos1d);
    auto reference = InverseSigmoid(reference_points);
    out_logits = class_embed->forward(output);
    auto h = output;
    for (std::size_t j = 0; j < bbox_embed->size(); ++j) {
      h = bbox_embed[j]->as<nn::LinearImpl>()->forward(h);
      if (j + 1 < bbox_embed->size()) h = torch::relu(h);
    }
    auto xy = h.index({"...", I::Slice(0, 2)}) + reference;
    auto wh = h.index({"...", I::Slice(2, 4)});
    out_boxes = torch::cat({xy, wh}, -1).sigmoid();
  }
  return {out_logits, out_boxes};
}

namespace {

class AnchorDetrImpl : public IModel {
 public:
  AnchorDetrImpl(std::string name, bool dc5) : name_(std::move(name)) {
    backbone_ = register_module("backbone",
                                ResNet(std::vector<int>{3, 4, 6, 3}, /*bottleneck=*/true, dc5));
    input_proj_ = register_module(
        "input_proj", nn::Sequential(nn::Conv2d(nn::Conv2dOptions(2048, cfg_.d_model, 1)),
                                     nn::GroupNorm(nn::GroupNormOptions(32, cfg_.d_model))));
    transformer_ = register_module("transformer", AnchorTransformer(cfg_));
  }

  Detections Forward(torch::Tensor images) override {
    auto feat = backbone_->forward(images);
    auto src = input_proj_->forward(feat);
    auto [logits, boxes] = transformer_->forward(src);
    Detections det;
    det.logits = logits;  // [B, nq, 91] sigmoid-focal
    det.boxes = boxes;    // [B, nq, 4] cxcywh in [0,1]
    return det;
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = name_;
    m.imgsz = 800;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_pattern * cfg_.num_position;
    m.focal = true;            // sigmoid head, top-100 over query x class
    m.imagenet_norm = true;    // ImageNet mean/std, aspect-preserving resize
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/megvii-research/AnchorDETR";
    return m;
  }

 private:
  AnchorTransformerConfig cfg_;
  std::string name_;
  ResNet backbone_{nullptr};
  nn::Sequential input_proj_{nullptr};
  AnchorTransformer transformer_{nullptr};
};

}  // namespace

void RegisterAnchorDetr() {
  Registry::Instance().Register(
      "anchor-detr", AnchorDetrImpl("anchor-detr", false).Meta(),
      [](const YAML::Node&) -> std::shared_ptr<IModel> {
        return std::make_shared<AnchorDetrImpl>("anchor-detr", /*dc5=*/false);
      });
  Registry::Instance().Register(
      "anchor-detr-dc5", AnchorDetrImpl("anchor-detr-dc5", true).Meta(),
      [](const YAML::Node&) -> std::shared_ptr<IModel> {
        return std::make_shared<AnchorDetrImpl>("anchor-detr-dc5", /*dc5=*/true);
      });
}

}  // namespace detr::models
