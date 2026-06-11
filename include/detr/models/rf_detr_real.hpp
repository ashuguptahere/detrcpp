// Copyright 2026 detrcpp authors. Apache-2.0.
//
// The faithful RF-DETR (roboflow/rf-detr): a DINOv2-windowed backbone + C2f
// projector feeding a two-stage single-scale deformable decoder. Reproduces the
// upstream architecture so official weights load and validate. Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include "detr/models/deform_attn.hpp"
#include "detr/models/deform_head.hpp"  // Mlp
#include "detr/models/dinov2_windowed.hpp"
#include "detr/models/model.hpp"
#include "detr/models/rf_detr_projector.hpp"

namespace detr::models {

// One RF-DETR decoder layer: separate-projection self-attn (8 heads) + MSDeformAttn
// cross-attn (16 heads, 1 level, 2 points) + a residual ReLU MLP, each post-normed.
struct RfDecoderLayerImpl : nn::Module {
  RfDecoderLayerImpl(int d, int sa_heads, int ca_heads, int n_levels, int n_points, int ffn);
  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& query_pos,
                        const torch::Tensor& reference_points, const torch::Tensor& memory,
                        const SpatialShapes& shapes);
  int sa_heads_;
  nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, o_proj{nullptr};
  nn::LayerNorm self_attn_layer_norm{nullptr}, cross_attn_layer_norm{nullptr}, layer_norm{nullptr};
  MSDeformAttn cross_attn{nullptr};
  nn::Linear fc1{nullptr}, fc2{nullptr};
};
TORCH_MODULE(RfDecoderLayer);

class RfDetrRealImpl : public IModel {
 public:
  // nano defaults; configurable for the other sizes.
  explicit RfDetrRealImpl(int num_classes = 91, int num_queries = 300, int imgsz = 384,
                          int vit_embed = 384, int vit_heads = 6, int patch = 16,
                          int num_windows = 2, int dec_layers = 2);

  Detections Forward(torch::Tensor images) override;
  ModelMeta Meta() const override;

  // The encoder-token indices chosen by two-stage query selection on the last
  // Forward, [B, num_queries]. Exposed for parity tests that align the query order
  // (torch.topk tie-breaks near-equal scores differently across backends).
  const torch::Tensor& LastTopkIndices() const { return topk_idx_; }

 private:
  int num_classes_, num_queries_, imgsz_, d_model_{256};
  torch::Tensor topk_idx_;

  Dinov2Windowed backbone_{nullptr};
  RfDetrProjector projector_{nullptr};
  // Two-stage query selection (group 0 only at inference).
  nn::Linear enc_output_{nullptr};
  nn::LayerNorm enc_output_norm_{nullptr};
  nn::Linear enc_out_class_{nullptr};
  Mlp enc_out_bbox_{nullptr};
  torch::Tensor reference_point_embed_;  // [num_queries*group, 4]
  torch::Tensor query_feat_;             // [num_queries*group, d]
  // Decoder.
  Mlp ref_point_head_{nullptr};
  nn::ModuleList decoder_{nullptr};
  nn::LayerNorm dec_layernorm_{nullptr};
  // Heads.
  nn::Linear class_embed_{nullptr};
  Mlp bbox_embed_{nullptr};
};

}  // namespace detr::models
