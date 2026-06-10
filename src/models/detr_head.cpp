// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/detr_head.hpp"

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "detr/models/pos_embed.hpp"

namespace detr::models {

namespace {

namespace nn = torch::nn;

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

  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& memory, const torch::Tensor& pos,
                        const torch::Tensor& query_pos) {
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

}  // namespace

DetrHead BuildDetrHead(torch::nn::Module& model, const DetrConfig& cfg) {
  DetrHead h;
  h.query_embed =
      model.register_module("query_embed", nn::Embedding(cfg.num_queries, cfg.hidden_dim));
  h.encoder = model.register_module("encoder", nn::ModuleList());
  for (int i = 0; i < cfg.enc_layers; ++i) {
    h.encoder->push_back(EncoderLayer(cfg.hidden_dim, cfg.nheads, cfg.dim_feedforward));
  }
  h.decoder = model.register_module("decoder", nn::ModuleList());
  for (int i = 0; i < cfg.dec_layers; ++i) {
    h.decoder->push_back(DecoderLayer(cfg.hidden_dim, cfg.nheads, cfg.dim_feedforward));
  }
  h.decoder_norm =
      model.register_module("decoder_norm", nn::LayerNorm(nn::LayerNormOptions({cfg.hidden_dim})));
  h.class_embed =
      model.register_module("class_embed", nn::Linear(cfg.hidden_dim, cfg.num_classes + 1));
  h.bbox_embed = model.register_module(
      "bbox_embed",
      nn::Sequential(nn::Linear(cfg.hidden_dim, cfg.hidden_dim), nn::Functional(torch::relu),
                     nn::Linear(cfg.hidden_dim, cfg.hidden_dim), nn::Functional(torch::relu),
                     nn::Linear(cfg.hidden_dim, 4)));
  return h;
}

Detections RunDetrHead(const DetrHead& head, torch::Tensor src, const DetrConfig& /*cfg*/) {
  const auto b = src.size(0);
  const auto d = src.size(1);
  const auto h = src.size(2);
  const auto w = src.size(3);

  auto pos = SinePos(b, d, h, w, src.options());
  auto src_seq = src.flatten(2).permute({2, 0, 1}).contiguous();  // [h*w, B, d]
  auto pos_seq = pos.flatten(2).permute({2, 0, 1}).contiguous();

  auto memory = src_seq;
  for (const auto& m : *head.encoder) {
    memory = m->as<EncoderLayerImpl>()->forward(memory, pos_seq);
  }

  auto query = head.query_embed->weight.unsqueeze(1).repeat({1, b, 1});  // [Q, B, d]
  auto tgt = torch::zeros_like(query);
  // Copy the holders (cheap shared handles) so forward() isn't called through a
  // const reference.
  auto decoder_norm = head.decoder_norm;
  auto class_embed = head.class_embed;
  auto bbox_embed = head.bbox_embed;

  // In training, collect every layer's (decoder-normed) output for deep
  // supervision. Each intermediate goes through DETR's shared decoder LayerNorm,
  // exactly like return_intermediate=True in the reference.
  const bool collect_aux = decoder_norm->is_training();
  std::vector<torch::Tensor> hs_layers;  // normed [B, Q, d] per decoder layer
  for (const auto& m : *head.decoder) {
    tgt = m->as<DecoderLayerImpl>()->forward(tgt, memory, pos_seq, query);
    if (collect_aux) {
      hs_layers.push_back(decoder_norm->forward(tgt).transpose(0, 1));
    }
  }

  Detections out;
  // Final output = last decoder layer after the decoder LayerNorm. When not
  // collecting aux this is the only norm/head application, so the inference path
  // is numerically identical to before.
  auto hs_final = collect_aux ? hs_layers.back() : decoder_norm->forward(tgt).transpose(0, 1);
  out.logits = class_embed->forward(hs_final);
  out.boxes = bbox_embed->forward(hs_final).sigmoid();
  // Auxiliary outputs: every layer except the last, through the shared heads.
  for (std::size_t i = 0; i + 1 < hs_layers.size(); ++i) {
    out.aux_logits.push_back(class_embed->forward(hs_layers[i]));
    out.aux_boxes.push_back(bbox_embed->forward(hs_layers[i]).sigmoid());
  }
  return out;
}

}  // namespace detr::models
