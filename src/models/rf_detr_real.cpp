// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/rf_detr_real.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace detr::models {

namespace {

constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

// Per-coordinate sinusoidal embedding (RF-DETR encode_sinusoidal_position_embedding).
// pos: [..., 4] in [0,1]; returns [..., 4*num_feats] with the DETR [y, x, w, h] swap.
torch::Tensor SineEmbed(const torch::Tensor& pos, std::int64_t num_feats) {
  auto dim_t = torch::arange(num_feats, pos.options().dtype(torch::kFloat32));
  dim_t = torch::pow(10000.0, 2 * torch::floor(dim_t / 2) / static_cast<double>(num_feats));
  std::vector<torch::Tensor> embs;
  for (const auto& c : pos.unbind(-1)) {  // x, y, w, h
    auto e = c.unsqueeze(-1) * kTwoPi / dim_t;  // [..., num_feats]
    auto even = e.slice(-1, 0, e.size(-1), 2).sin();
    auto odd = e.slice(-1, 1, e.size(-1), 2).cos();
    embs.push_back(torch::stack({even, odd}, -1).flatten(-2, -1));
  }
  std::swap(embs[0], embs[1]);  // [y, x, w, h]
  return torch::cat(embs, -1);
}

// new = (delta[:2]*ref[2:] + ref[:2], delta[2:].exp()*ref[2:]) — RF-DETR bbox reparam.
torch::Tensor RefineBboxes(const torch::Tensor& ref, const torch::Tensor& delta) {
  auto cxcy = delta.narrow(-1, 0, 2) * ref.narrow(-1, 2, 2) + ref.narrow(-1, 0, 2);
  auto wh = delta.narrow(-1, 2, 2).exp() * ref.narrow(-1, 2, 2);
  return torch::cat({cxcy, wh}, -1);
}

// Grid-anchor proposals for one level: [1, H*W, 4] cxcywh, centers (i+.5)/size,
// wh = 0.05 * 2^level (coarser levels get larger anchors).
torch::Tensor GridProposals(std::int64_t h, std::int64_t w, int level,
                            const torch::TensorOptions& opts) {
  auto gy = torch::arange(h, opts).unsqueeze(1).expand({h, w});
  auto gx = torch::arange(w, opts).unsqueeze(0).expand({h, w});
  auto grid = torch::stack({gx, gy}, -1);  // [h,w,2] (x,y)
  grid = (grid + 0.5) / torch::tensor({static_cast<float>(w), static_cast<float>(h)}, opts);
  auto wh = torch::full({h, w, 2}, 0.05 * std::pow(2.0, level), opts);
  return torch::cat({grid, wh}, -1).view({1, h * w, 4});
}

}  // namespace

RfDecoderLayerImpl::RfDecoderLayerImpl(int d, int sa_heads, int ca_heads, int n_levels,
                                       int n_points, int ffn)
    : sa_heads_(sa_heads) {
  q_proj = register_module("q_proj", nn::Linear(d, d));
  k_proj = register_module("k_proj", nn::Linear(d, d));
  v_proj = register_module("v_proj", nn::Linear(d, d));
  o_proj = register_module("o_proj", nn::Linear(d, d));
  self_attn_layer_norm = register_module("self_attn_layer_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
  cross_attn = register_module("cross_attn", MSDeformAttn(d, n_levels, ca_heads, n_points));
  cross_attn_layer_norm = register_module("cross_attn_layer_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
  fc1 = register_module("fc1", nn::Linear(d, ffn));
  fc2 = register_module("fc2", nn::Linear(ffn, d));
  layer_norm = register_module("layer_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
}

torch::Tensor RfDecoderLayerImpl::forward(torch::Tensor tgt, const torch::Tensor& query_pos,
                                          const torch::Tensor& reference_points,
                                          const torch::Tensor& memory, const SpatialShapes& shapes) {
  const auto b = tgt.size(0);
  const auto t = tgt.size(1);
  const auto c = tgt.size(2);
  const auto hd = c / sa_heads_;
  // Self-attention: q=k=tgt+pos, v=tgt.
  auto qk = tgt + query_pos;
  auto split = [&](const torch::Tensor& y) { return y.view({b, t, sa_heads_, hd}).transpose(1, 2); };
  auto qh = split(q_proj->forward(qk));
  auto kh = split(k_proj->forward(qk));
  auto vh = split(v_proj->forward(tgt));
  auto sc = torch::matmul(qh, kh.transpose(-1, -2)) / std::sqrt(static_cast<double>(hd));
  auto ctx = torch::matmul(sc.softmax(-1), vh).transpose(1, 2).contiguous().view({b, t, c});
  tgt = self_attn_layer_norm->forward(tgt + o_proj->forward(ctx));
  // Deformable cross-attention (query carries the position embedding).
  auto ca = cross_attn->forward(tgt + query_pos, reference_points, memory, shapes);
  tgt = cross_attn_layer_norm->forward(tgt + ca);
  // Residual MLP (decoder_activation_function = relu).
  auto m = tgt + fc2->forward(torch::relu(fc1->forward(tgt)));
  return layer_norm->forward(m);
}

RfDetrRealImpl::RfDetrRealImpl(RfDetrRealConfig cfg) : cfg_(std::move(cfg)) {
  d_model_ = cfg_.d_model;
  const int d = d_model_;
  const int c = cfg_.num_classes;
  const int n_levels = cfg_.scale_factors.empty() ? 1 : static_cast<int>(cfg_.scale_factors.size());
  const int pe = cfg_.pe_grid > 0 ? cfg_.pe_grid : cfg_.imgsz / cfg_.patch;
  if (cfg_.backbone == RfDetrRealConfig::kLwDetrViT) {
    backbone_lw_ = register_module(
        "backbone", LwDetrViT(cfg_.vit_embed, cfg_.vit_depth, cfg_.vit_heads, cfg_.patch,
                              cfg_.num_windows, pe, cfg_.out_indices, cfg_.window_blocks));
  } else {
    backbone_dino_ = register_module(
        "backbone", Dinov2Windowed(cfg_.vit_embed, cfg_.vit_depth, cfg_.vit_heads, cfg_.patch,
                                   cfg_.num_windows, pe, 0, cfg_.out_indices, cfg_.window_blocks));
  }
  const int n_feats = static_cast<int>(cfg_.out_indices.size());  // 4 (RF-DETR) or 3 (LW-DETR-tiny)
  if (cfg_.scale_factors.empty()) {
    projector_ = register_module(
        "projector", RfDetrProjector(n_feats, cfg_.vit_embed, d, 3, cfg_.projector_batchnorm));
  } else {
    ms_projector_ = register_module(
        "projector", LwDetrMultiScaleProjector(n_feats, cfg_.vit_embed, d, 3, cfg_.scale_factors));
  }
  enc_output_ = register_module("enc_output", nn::Linear(d, d));
  enc_output_norm_ = register_module("enc_output_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
  enc_out_class_ = register_module("enc_out_class", nn::Linear(d, c));
  enc_out_bbox_ = register_module("enc_out_bbox", Mlp(d, d, 4, 3));
  reference_point_embed_ =
      register_parameter("reference_point_embed", torch::zeros({cfg_.num_queries, 4}));
  query_feat_ = register_parameter("query_feat", torch::zeros({cfg_.num_queries, d}));
  ref_point_head_ = register_module("ref_point_head", Mlp(2 * d, d, d, 2));
  decoder_ = register_module("decoder", nn::ModuleList());
  const int sa_heads = d / 32;  // head_dim 32 (8 @ d=256, 12 @ d=384)
  const int ca_heads = d / 16;  // head_dim 16 (16 @ d=256, 24 @ d=384)
  for (int i = 0; i < cfg_.dec_layers; ++i) {
    decoder_->push_back(RfDecoderLayer(d, sa_heads, ca_heads, n_levels, cfg_.n_points, 2048));
  }
  dec_layernorm_ = register_module("dec_layernorm", nn::LayerNorm(nn::LayerNormOptions({d})));
  class_embed_ = register_module("class_embed", nn::Linear(d, c));
  bbox_embed_ = register_module("bbox_embed", Mlp(d, d, 4, 3));
}

Detections RfDetrRealImpl::Forward(torch::Tensor images) {
  auto feats = backbone_lw_ ? backbone_lw_->forward(images)   // N x [B, vit_embed, h, w]
                            : backbone_dino_->forward(images);
  // Projector -> one feature map per decoder level (single-scale: just one).
  auto levels = ms_projector_ ? ms_projector_->forward(feats)
                              : std::vector<torch::Tensor>{projector_->forward(feats)};
  const auto b = levels[0].size(0);
  const int n_levels = static_cast<int>(levels.size());
  SpatialShapes shapes;
  std::vector<torch::Tensor> mem_parts, prop_parts;
  for (int l = 0; l < n_levels; ++l) {
    const auto h = levels[static_cast<std::size_t>(l)].size(2);
    const auto w = levels[static_cast<std::size_t>(l)].size(3);
    shapes.emplace_back(h, w);
    mem_parts.push_back(levels[static_cast<std::size_t>(l)].flatten(2).transpose(1, 2));  // [B,hw,d]
    prop_parts.push_back(GridProposals(h, w, l, levels[static_cast<std::size_t>(l)].options()));
  }
  auto memory = torch::cat(mem_parts, 1).contiguous();                  // [B, sum(hw), d]

  // Two-stage query selection (group 0). Grid anchors whose coords fall outside
  // (0.01, 0.99) are "invalid": their object-query is zeroed and their class score is
  // forced to -inf so they're never selected (matters when a level's grid reaches the
  // image edge — e.g. LW-DETR's P3 80x80; a no-op for the 40x40 single-scale levels).
  auto proposals = torch::cat(prop_parts, 1).expand({b, -1, -1}).contiguous();  // [B,sum_hw,4]
  auto invalid = ((proposals > 0.01) & (proposals < 0.99)).all(-1, /*keepdim=*/true).logical_not();
  auto obj_query = memory.masked_fill(invalid, 0.0);                           // [B,hw,d]
  proposals = proposals.masked_fill(invalid, 0.0);
  auto oq = enc_output_norm_->forward(enc_output_->forward(obj_query));        // [B,hw,d]
  auto enc_class =
      enc_out_class_->forward(oq).masked_fill(invalid, -std::numeric_limits<float>::infinity());
  auto enc_coord = RefineBboxes(proposals, enc_out_bbox_->forward(oq));        // [B,hw,4]
  auto topk =
      std::get<1>(std::get<0>(enc_class.max(-1)).topk(cfg_.num_queries, 1));   // [B,nq]
  topk_idx_ = topk;
  auto gi = topk.unsqueeze(-1);
  auto topk_coords = enc_coord.gather(1, gi.expand({b, cfg_.num_queries, 4})); // [B,nq,4]
  auto rpe = reference_point_embed_.unsqueeze(0).expand({b, -1, -1});
  auto reference_points = RefineBboxes(topk_coords, rpe);                      // [B,nq,4]
  auto tgt = query_feat_.unsqueeze(0).expand({b, -1, -1}).contiguous();        // [B,nq,d]

  // Decoder (fixed reference points; query position from the sine embed). The 4D
  // reference points are shared across levels — replicate for the deformable attn.
  auto ref_input = reference_points.unsqueeze(2);                              // [B,nq,1,4]
  if (n_levels > 1) {
    ref_input = ref_input.expand({b, cfg_.num_queries, n_levels, 4}).contiguous();
  }
  auto query_pos = ref_point_head_->forward(SineEmbed(reference_points, d_model_ / 2));
  for (const auto& layer : *decoder_) {
    tgt = layer->as<RfDecoderLayerImpl>()->forward(tgt, query_pos, ref_input, memory, shapes);
  }
  auto hidden = dec_layernorm_->forward(tgt);

  Detections det;
  det.logits = class_embed_->forward(hidden);                                 // [B,nq,C]
  det.boxes = RefineBboxes(reference_points, bbox_embed_->forward(hidden));    // [B,nq,4] cxcywh
  return det;
}

ModelMeta RfDetrRealImpl::Meta() const {
  ModelMeta m;
  m.name = cfg_.name;
  m.imgsz = cfg_.imgsz;
  m.num_classes = cfg_.num_classes;
  m.num_queries = cfg_.num_queries;
  m.focal = true;                       // sigmoid/focal head, no no-object slot
  m.imagenet_norm = cfg_.imagenet_norm;  // [0,1] then ImageNet mean/std, square resize
  m.license = "Apache-2.0";
  m.upstream = cfg_.upstream;
  return m;
}

// Maps a native Atten4Vis/LW-DETR checkpoint onto our RfDetrReal(LW-DETR) tree. RF-DETR
// (DINOv2 backbone) shares this graph but is left identity (its native weights aren't
// reachable to verify here). The fused-qkv / fused-in_proj projections are split at load
// (SplitRows); the group-DETR query embeddings are sliced to num_queries (SliceRows).
weights::WeightRemapper RfDetrRealImpl::UpstreamRemapper() const {
  weights::WeightRemapper r;
  if (cfg_.backbone != RfDetrRealConfig::kLwDetrViT) return r;  // RF-DETR: separately.

  // --- Backbone (ViT): backbone.0.encoder.* -> backbone.*; attn.* flattened. ---
  r.ReplaceRegex("^backbone\\.0\\.encoder\\.patch_embed\\.proj\\.", "backbone.patch_embed.")
      .ReplaceRegex("^backbone\\.0\\.encoder\\.pos_embed$", "backbone.pos_embed")
      .ReplaceRegex("^backbone\\.0\\.encoder\\.blocks\\.", "backbone.blocks.")
      .ReplaceRegex("(backbone\\.blocks\\.[0-9]+)\\.attn\\.q_bias$", "$1.q.bias")
      .ReplaceRegex("(backbone\\.blocks\\.[0-9]+)\\.attn\\.v_bias$", "$1.v.bias")
      .ReplaceRegex("(backbone\\.blocks\\.[0-9]+)\\.attn\\.proj\\.", "$1.o.")
      .ReplaceRegex("(backbone\\.blocks\\.[0-9]+)\\.mlp\\.fc1\\.", "$1.fc1.")
      .ReplaceRegex("(backbone\\.blocks\\.[0-9]+)\\.mlp\\.fc2\\.", "$1.fc2.")
      .ReplaceRegex("(backbone\\.blocks\\.[0-9]+)\\.gamma_1$", "$1.gamma1")
      .ReplaceRegex("(backbone\\.blocks\\.[0-9]+)\\.gamma_2$", "$1.gamma2")
      .SplitRows("^(backbone\\.blocks\\.[0-9]+)\\.attn\\.qkv\\.weight$",
                 {"$1.q.weight", "$1.k.weight", "$1.v.weight"});

  // --- Projector: single-scale (tiny/small/medium) or multi-scale (large/xlarge). ---
  if (cfg_.scale_factors.empty()) {
    r.ReplaceRegex("^backbone\\.0\\.projector\\.stages\\.0\\.0\\.", "projector.stage.")
        .ReplaceRegex("^backbone\\.0\\.projector\\.stages\\.0\\.1\\.", "projector.norm.");
  } else {
    r.ReplaceRegex("^backbone\\.0\\.projector\\.stages_sampling\\.([0-9]+)\\.([0-9]+)\\.",
                   "projector.scale_layers.$1.sampling.$2.layers.")
        .ReplaceRegex("^backbone\\.0\\.projector\\.stages\\.([0-9]+)\\.0\\.",
                      "projector.scale_layers.$1.stage.")
        .ReplaceRegex("^backbone\\.0\\.projector\\.stages\\.([0-9]+)\\.1\\.",
                      "projector.scale_layers.$1.norm.");
  }

  // --- Two-stage heads + decoder norm + query selection. The enc heads are group-DETR
  // ModuleLists (one per training group); inference uses group 0, drop the rest. ---
  r.Drop("^transformer\\.enc_output\\.[1-9]")
      .Drop("^transformer\\.enc_output_norm\\.[1-9]")
      .Drop("^transformer\\.enc_out_class_embed\\.[1-9]")
      .Drop("^transformer\\.enc_out_bbox_embed\\.[1-9]")
      .ReplaceRegex("^transformer\\.enc_output\\.0\\.", "enc_output.")
      .ReplaceRegex("^transformer\\.enc_output_norm\\.0\\.", "enc_output_norm.")
      .ReplaceRegex("^transformer\\.enc_out_class_embed\\.0\\.", "enc_out_class.")
      .ReplaceRegex("^transformer\\.enc_out_bbox_embed\\.0\\.", "enc_out_bbox.")
      .ReplaceRegex("^transformer\\.decoder\\.norm\\.", "dec_layernorm.")
      .ReplaceRegex("^transformer\\.decoder\\.ref_point_head\\.", "ref_point_head.")
      .ReplaceRegex("^refpoint_embed\\.weight$", "reference_point_embed")
      .ReplaceRegex("^query_feat\\.weight$", "query_feat")
      // group-DETR stores [num_groups*nq, *]; inference uses the leading num_queries rows.
      .SliceRows("^reference_point_embed$", cfg_.num_queries)
      .SliceRows("^query_feat$", cfg_.num_queries);

  // --- Decoder layers: transformer.decoder.layers.N -> decoder.N; in_proj split. ---
  r.ReplaceRegex("^transformer\\.decoder\\.layers\\.", "decoder.")
      .ReplaceRegex("(decoder\\.[0-9]+)\\.self_attn\\.out_proj\\.", "$1.o_proj.")
      .ReplaceRegex("(decoder\\.[0-9]+)\\.norm1\\.", "$1.self_attn_layer_norm.")
      .ReplaceRegex("(decoder\\.[0-9]+)\\.norm2\\.", "$1.cross_attn_layer_norm.")
      .ReplaceRegex("(decoder\\.[0-9]+)\\.norm3\\.", "$1.layer_norm.")
      .ReplaceRegex("(decoder\\.[0-9]+)\\.linear1\\.", "$1.fc1.")
      .ReplaceRegex("(decoder\\.[0-9]+)\\.linear2\\.", "$1.fc2.")
      .SplitRows("^(decoder\\.[0-9]+)\\.self_attn\\.in_proj_weight$",
                 {"$1.q_proj.weight", "$1.k_proj.weight", "$1.v_proj.weight"})
      .SplitRows("^(decoder\\.[0-9]+)\\.self_attn\\.in_proj_bias$",
                 {"$1.q_proj.bias", "$1.k_proj.bias", "$1.v_proj.bias"});
  return r;
}

}  // namespace detr::models
