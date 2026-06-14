// Copyright 2026 detrcpp authors. Apache-2.0.
//
// The faithful RF-DETR / LW-DETR family: a windowed-ViT backbone (DINOv2-windowed
// for RF-DETR, plain CAEv2-ViT for LW-DETR) + C2f projector feeding a two-stage
// single-scale deformable decoder. RF-DETR is built on LW-DETR, so the projector,
// decoder and heads are identical — only the backbone and decoder depth differ.
// Reproduces the upstream architectures so official weights load and validate.
// Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "detr/models/deform_attn.hpp"
#include "detr/models/deform_head.hpp"  // Mlp
#include "detr/models/dinov2_windowed.hpp"
#include "detr/models/lw_detr_vit.hpp"
#include "detr/models/model.hpp"
#include "detr/models/rf_detr_projector.hpp"

namespace detr::models {

// Configures a member of the RF-DETR / LW-DETR family. Defaults are RF-DETR-Nano.
struct RfDetrRealConfig {
  enum BackboneKind { kDinov2Windowed, kLwDetrViT };
  std::string name = "rf-detr-nano";
  std::string upstream = "https://github.com/roboflow/rf-detr";
  int num_classes = 91, num_queries = 300, imgsz = 384, dec_layers = 2;
  int d_model = 256, n_points = 2;
  std::vector<double> scale_factors = {};  // empty => single-scale; e.g. {2.0,0.5} multi
  bool imagenet_norm = true;
  BackboneKind backbone = kDinov2Windowed;
  int vit_embed = 384, vit_depth = 12, vit_heads = 6, patch = 16, num_windows = 2;
  int pe_grid = 0;  // pos-embed grid; 0 => imgsz/patch (DINOv2), else explicit (LW-DETR)
  bool projector_batchnorm = false;  // C2f conv norm: false=LayerNorm (RF), true=BN (LW)
  std::vector<int> out_indices = {3, 6, 9, 12};
  std::vector<int> window_blocks = {0, 1, 2, 4, 5, 7, 8, 10, 11};
};

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
  explicit RfDetrRealImpl(RfDetrRealConfig cfg = {});

  Detections Forward(torch::Tensor images) override;
  ModelMeta Meta() const override;

  // The encoder-token indices chosen by two-stage query selection on the last
  // Forward, [B, num_queries]. Exposed for parity tests that align the query order
  // (torch.topk tie-breaks near-equal scores differently across backends).
  const torch::Tensor& LastTopkIndices() const { return topk_idx_; }

 private:
  RfDetrRealConfig cfg_;
  int d_model_{256};
  torch::Tensor topk_idx_;

  Dinov2Windowed backbone_dino_{nullptr};  // RF-DETR backbone
  LwDetrViT backbone_lw_{nullptr};         // LW-DETR backbone (exactly one is built)
  RfDetrProjector projector_{nullptr};                // single-scale
  LwDetrMultiScaleProjector ms_projector_{nullptr};   // multi-scale (LW-DETR L/X)
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
