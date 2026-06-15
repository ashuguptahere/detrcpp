// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/deform_head.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace detr::models {

namespace {

namespace nn = torch::nn;

torch::Tensor InverseSigmoid(torch::Tensor x, double eps = 1e-5) {
  x = x.clamp(0, 1);
  return torch::log(x.clamp_min(eps) / (1 - x).clamp_min(eps));
}

// Grid-center anchors in inverse-sigmoid space: [1, Sum(H*W), 4].
torch::Tensor GenerateAnchors(const SpatialShapes& shapes, const torch::TensorOptions& opts) {
  std::vector<torch::Tensor> anchors;
  int lvl = 0;
  for (const auto& [h, w] : shapes) {
    auto grid = torch::meshgrid({torch::arange(h, opts), torch::arange(w, opts)}, "ij");
    auto xy = torch::stack({grid[1], grid[0]}, -1);
    auto wht = torch::tensor({static_cast<double>(w), static_cast<double>(h)}, opts);
    auto xyn = (xy.unsqueeze(0) + 0.5) / wht;
    auto wh = torch::ones_like(xyn) * (0.05 * std::pow(2.0, lvl));
    anchors.push_back(torch::cat({xyn, wh}, -1).reshape({1, h * w, 4}));
    ++lvl;
  }
  auto a = torch::cat(anchors, 1);
  auto valid = ((a > 1e-2) * (a < 1 - 1e-2)).all(-1, true);
  a = torch::log(a / (1 - a));
  return torch::where(valid.expand_as(a), a, torch::full_like(a, 1e9));
}

// Deformable decoder layer: self-attn + deformable cross-attn (4D refs) + FFN.
struct DeformDecoderLayerImpl : nn::Module {
  nn::MultiheadAttention self_attn{nullptr};
  MSDeformAttn cross_attn{nullptr};
  nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
  nn::Linear linear1{nullptr}, linear2{nullptr};
  bool silu_;
  DeformDecoderLayerImpl(int d, int levels, int heads, int points, int ff,
                         bool discrete_sample = false, bool silu = false)
      : silu_(silu) {
    self_attn = register_module(
        "self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, heads).dropout(0.0)));
    cross_attn =
        register_module("cross_attn", MSDeformAttn(d, levels, heads, points, discrete_sample));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm3 = register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({d})));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
  }
  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& query_pos, const torch::Tensor& ref,
                        const torch::Tensor& memory, const SpatialShapes& shapes,
                        const torch::Tensor& self_attn_mask = {}) {
    auto q = (tgt + query_pos).transpose(0, 1);
    // self_attn_mask is an ADDITIVE [L,L] float mask (torch C++ MHA adds it pre-
    // softmax); undefined => no mask (rt-detr/rf/dino-plain byte-identical).
    auto sa = std::get<0>(self_attn->forward(q, q, tgt.transpose(0, 1), /*key_padding_mask=*/{},
                                             /*need_weights=*/true, /*attn_mask=*/self_attn_mask))
                  .transpose(0, 1);
    tgt = norm1->forward(tgt + sa);
    auto ca = cross_attn->forward(tgt + query_pos, ref, memory, shapes);
    tgt = norm2->forward(tgt + ca);
    auto h = linear1->forward(tgt);
    auto ff = linear2->forward(silu_ ? torch::silu(h) : torch::relu(h));
    return norm3->forward(tgt + ff);
  }
};
TORCH_MODULE(DeformDecoderLayer);

}  // namespace

MlpImpl::MlpImpl(int in, int hidden, int out, int n, bool silu) : n_(n), silu_(silu) {
  layers = register_module("layers", nn::ModuleList());
  int prev = in;
  for (int i = 0; i < n; ++i) {
    layers->push_back(nn::Linear(prev, (i + 1 == n) ? out : hidden));
    prev = (i + 1 == n) ? out : hidden;
  }
}

torch::Tensor MlpImpl::forward(torch::Tensor x) {
  for (int i = 0; i < n_; ++i) {
    x = layers[static_cast<std::size_t>(i)]->as<nn::LinearImpl>()->forward(x);
    if (i + 1 < n_) {
      x = silu_ ? torch::silu(x) : torch::relu(x);
    }
  }
  return x;
}

DeformDetectHead BuildDeformDetectHead(nn::Module& m, int d, int levels, int heads, int points,
                                       int ff, int dec_layers, int num_classes, int num_queries,
                                       bool discrete_sample, bool deim) {
  DeformDetectHead h;
  h.num_levels = levels;
  h.num_queries = num_queries;
  h.enc_output = m.register_module("enc_output", nn::Linear(d, d));
  h.enc_output_norm =
      m.register_module("enc_output_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
  h.enc_score_head = m.register_module("enc_score_head", nn::Linear(d, num_classes));
  h.enc_bbox_head = m.register_module("enc_bbox_head", Mlp(d, d, 4, 3, deim));
  // DEIM-RT-DETRv2 uses a 3-layer query_pos_head MLP(4,d,d,3); RT-DETR uses MLP(4,2d,d,2).
  h.query_pos_head = m.register_module(
      "query_pos_head", deim ? Mlp(4, d, d, 3, true) : Mlp(4, 2 * d, d, 2));
  h.decoder = m.register_module("decoder", nn::ModuleList());
  h.dec_score = m.register_module("dec_score_head", nn::ModuleList());
  h.dec_bbox = m.register_module("dec_bbox_head", nn::ModuleList());
  for (int i = 0; i < dec_layers; ++i) {
    h.decoder->push_back(DeformDecoderLayer(d, levels, heads, points, ff, discrete_sample, deim));
    h.dec_score->push_back(nn::Linear(d, num_classes));
    h.dec_bbox->push_back(Mlp(d, d, 4, 3, deim));
  }
  return h;
}

namespace {

// Query selection + the deformable decoder loop. Optionally prepends CDN
// denoising queries (DINO-CDN) under cdn.attn_mask and splits the outputs.
Detections RunCore(const DeformDetectHead& head, torch::Tensor memory, const SpatialShapes& shapes,
                   const DeformCdn& cdn, DenoisingOut& dn_out) {
  const auto b = memory.size(0);
  const auto d = memory.size(2);
  const auto nq = head.num_queries;

  // Copy the holders (cheap shared handles) so non-const forward() isn't called
  // through the const head reference.
  auto enc_output = head.enc_output;
  auto enc_output_norm = head.enc_output_norm;
  auto enc_score_head = head.enc_score_head;
  auto enc_bbox_head = head.enc_bbox_head;
  auto query_pos_head = head.query_pos_head;
  auto decoder = head.decoder;
  auto dec_score = head.dec_score;
  auto dec_bbox = head.dec_bbox;

  auto anchors = GenerateAnchors(shapes, memory.options());
  auto out_mem = enc_output_norm->forward(enc_output->forward(memory));
  auto enc_class = enc_score_head->forward(out_mem);
  auto enc_coord = enc_bbox_head->forward(out_mem) + anchors;

  auto topk = std::get<1>(std::get<0>(enc_class.max(-1)).topk(nq, 1));
  auto gi = topk.unsqueeze(-1);
  auto ref_unact = enc_coord.gather(1, gi.expand({b, nq, 4})).detach();
  auto tgt = out_mem.gather(1, gi.expand({b, nq, d})).detach();
  auto ref = ref_unact.sigmoid();

  // In training, keep each layer's prediction for deep supervision.
  const bool collect_aux = enc_output->is_training();
  std::int64_t num_dn = 0;
  torch::Tensor add_mask;  // undefined unless CDN is active
  if (cdn.active && collect_aux) {            // training-only (matches dab_detr)
    num_dn = cdn.num_dn;
    tgt = torch::cat({cdn.dn_tgt, tgt}, 1);   // [B, num_dn+nq, d]
    ref = torch::cat({cdn.dn_ref, ref}, 1);   // [B, num_dn+nq, 4]
    const std::int64_t L = num_dn + nq;
    add_mask = torch::zeros({L, L}, memory.options())
                   .masked_fill(cdn.attn_mask, -std::numeric_limits<float>::infinity());
  }
  const std::int64_t Lq = ref.size(1);  // == nq when no CDN

  const int n = static_cast<int>(decoder->size());
  Detections det;
  torch::Tensor logits;
  torch::Tensor boxes;
  for (int i = 0; i < n; ++i) {
    auto ref_input = ref.unsqueeze(2).expand({b, Lq, head.num_levels, 4});
    auto query_pos = query_pos_head->forward(ref);
    tgt = decoder[static_cast<std::size_t>(i)]->as<DeformDecoderLayerImpl>()->forward(
        tgt, query_pos, ref_input, memory, shapes, add_mask);
    auto bbox =
        (dec_bbox[static_cast<std::size_t>(i)]->as<MlpImpl>()->forward(tgt) + InverseSigmoid(ref))
            .sigmoid();
    logits = dec_score[static_cast<std::size_t>(i)]->as<nn::LinearImpl>()->forward(tgt);
    boxes = bbox;
    if (collect_aux && i + 1 < n) {
      if (num_dn > 0) {
        dn_out.dn_aux_logits.push_back(logits.narrow(1, 0, num_dn));
        dn_out.dn_aux_boxes.push_back(boxes.narrow(1, 0, num_dn));
        det.aux_logits.push_back(logits.narrow(1, num_dn, nq));
        det.aux_boxes.push_back(boxes.narrow(1, num_dn, nq));
      } else {
        det.aux_logits.push_back(logits);
        det.aux_boxes.push_back(boxes);
      }
    }
    ref = bbox.detach();
  }

  if (num_dn > 0) {
    det.logits = logits.narrow(1, num_dn, nq);
    det.boxes = boxes.narrow(1, num_dn, nq);
    dn_out.active = true;
    dn_out.num_dn = static_cast<int>(num_dn);
    dn_out.dn_logits = logits.narrow(1, 0, num_dn);
    dn_out.dn_boxes = boxes.narrow(1, 0, num_dn);
  } else {
    det.logits = logits;
    det.boxes = boxes;
  }
  return det;
}

}  // namespace

Detections RunDeformDetectHead(const DeformDetectHead& head, torch::Tensor memory,
                               const SpatialShapes& shapes) {
  DenoisingOut sink;  // discarded
  return RunCore(head, std::move(memory), shapes, DeformCdn{}, sink);
}

Detections RunDeformDetectHead(const DeformDetectHead& head, torch::Tensor memory,
                               const SpatialShapes& shapes, const DeformCdn& cdn,
                               DenoisingOut& dn_out) {
  dn_out = DenoisingOut{};
  return RunCore(head, std::move(memory), shapes, cdn, dn_out);
}

}  // namespace detr::models
