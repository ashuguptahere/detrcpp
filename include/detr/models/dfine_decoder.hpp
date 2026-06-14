// Copyright 2026 detrcpp authors. Apache-2.0.
//
// D-FINE decoder + heads (DFINETransformer): two-stage query selection over the neck
// memory then a deformable decoder implementing Fine-grained Distribution Refinement
// (FDR). Each layer predicts, per box edge, a softmax distribution over reg_max+1 bins
// that is integrated against a non-uniform weighting W(n) into edge distances and
// decoded to a box (distance2bbox); the distributions refine residually across layers,
// and an LQE head adds a localization-quality term to the class scores. Inference only
// (denoising / GO-LSD distillation are training-only and omitted). Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace detr::models {

namespace nn = torch::nn;

using DfShapes = std::vector<std::pair<std::int64_t, std::int64_t>>;

// A simple MLP (D-FINE `MLP`): `num_layers` Linear layers, ReLU between them.
struct DfMLPImpl : nn::Module {
  DfMLPImpl(int in_dim, int hidden_dim, int out_dim, int num_layers);
  torch::Tensor forward(torch::Tensor x);
  nn::ModuleList layers{nullptr};
  int num_layers_;
};
TORCH_MODULE(DfMLP);

// enc_output: Linear projection + LayerNorm (named "proj"/"norm").
struct DfEncOutputImpl : nn::Module {
  explicit DfEncOutputImpl(int d);
  torch::Tensor forward(torch::Tensor x);
  nn::Linear proj{nullptr};
  nn::LayerNorm norm{nullptr};
};
TORCH_MODULE(DfEncOutput);

// D-FINE deformable cross-attention: only sampling_offsets + attention_weights (value is
// pre-split, no value/output projection). 4D reference -> per-edge box-scaled sampling.
struct DfMSDeformImpl : nn::Module {
  DfMSDeformImpl(int d, int nhead, int num_levels, std::vector<int> num_points);
  // value: per-level [B, nhead, head_dim, H*W]; ref: [B, nq, 1, 4].
  torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& ref,
                        const std::vector<torch::Tensor>& value, const DfShapes& shapes);
  int nhead_, num_levels_;
  std::vector<int> num_points_;
  int total_points_;
  double offset_scale_{0.5};
  torch::Tensor num_points_scale_;  // [sum(num_points)]
  nn::Linear sampling_offsets{nullptr};
  nn::Linear attention_weights{nullptr};
};
TORCH_MODULE(DfMSDeform);

// Gating fusion of self-attn output and cross-attn output, post-normed.
struct DfGateImpl : nn::Module {
  explicit DfGateImpl(int d);
  torch::Tensor forward(const torch::Tensor& x1, const torch::Tensor& x2);
  nn::Linear gate{nullptr};
  nn::LayerNorm norm{nullptr};
};
TORCH_MODULE(DfGate);

// One decoder layer: self-attn (post-norm) -> deformable cross-attn (gated) -> FFN.
struct DfDecLayerImpl : nn::Module {
  DfDecLayerImpl(int d, int nhead, int ffn, int num_levels, std::vector<int> num_points);
  torch::Tensor forward(torch::Tensor target, const torch::Tensor& ref,
                        const std::vector<torch::Tensor>& value, const DfShapes& shapes,
                        const torch::Tensor& query_pos);
  nn::MultiheadAttention self_attn{nullptr};
  nn::LayerNorm norm1{nullptr}, norm3{nullptr};
  DfMSDeform cross_attn{nullptr};
  DfGate gateway{nullptr};
  nn::Linear linear1{nullptr}, linear2{nullptr};
};
TORCH_MODULE(DfDecLayer);

// Location Quality Estimator: a small MLP over the top-k corner-distribution stats,
// added to the class scores.
struct DfLQEImpl : nn::Module {
  DfLQEImpl(int k, int hidden_dim, int num_layers, int reg_max);
  torch::Tensor forward(const torch::Tensor& scores, const torch::Tensor& pred_corners);
  int k_, reg_max_;
  DfMLP reg_conf{nullptr};
};
TORCH_MODULE(DfLQE);

// The deformable decoder stack: per-layer DfDecLayer + LQE. The FDR refinement loop
// itself lives in DFINETransformer (it needs the shared bbox/score heads).
struct DfDecoderStackImpl : nn::Module {
  DfDecoderStackImpl(int num_layers, int d, int nhead, int ffn, int num_levels,
                     std::vector<int> num_points, int reg_max);
  nn::ModuleList layers{nullptr};      // DfDecLayer
  nn::ModuleList lqe_layers{nullptr};  // DfLQE
};
TORCH_MODULE(DfDecoderStack);

// Configures a DFINETransformer (the D-FINE decoder + heads). Defaults are D-FINE-L.
struct DfTransformerConfig {
  int num_classes = 80;
  int hidden_dim = 256;
  int num_queries = 300;
  std::vector<int> feat_strides = {8, 16, 32};
  int num_levels = 3;
  std::vector<int> num_points = {3, 6, 3};
  int nhead = 8;
  int num_layers = 6;
  int dim_feedforward = 1024;
  int reg_max = 32;
  double reg_scale = 4.0;
  double up = 0.5;
  double eps = 1e-2;
};

struct DFINETransformerImpl : nn::Module {
  explicit DFINETransformerImpl(DfTransformerConfig cfg);
  // feats: the neck outputs (num_levels x [B, hidden_dim, H, W]). Returns {logits, boxes}:
  // logits [B, num_queries, num_classes] (pre-sigmoid), boxes [B, num_queries, 4] cxcywh.
  std::pair<torch::Tensor, torch::Tensor> forward(const std::vector<torch::Tensor>& feats);
  // Encoder-token indices chosen by query selection on the last Forward, [B, num_queries].
  const torch::Tensor& LastTopkIndices() const { return topk_idx_; }

  DfTransformerConfig cfg_;
  torch::Tensor project_;  // weighting function W(n), [reg_max+1]
  torch::Tensor topk_idx_;

  DfEncOutput enc_output{nullptr};
  nn::Linear enc_score_head{nullptr};
  DfMLP enc_bbox_head{nullptr};
  DfMLP query_pos_head{nullptr};
  DfMLP pre_bbox_head{nullptr};
  nn::ModuleList dec_score_head{nullptr};  // per layer: Linear -> num_classes
  nn::ModuleList dec_bbox_head{nullptr};   // per layer: DfMLP -> 4*(reg_max+1)
  DfDecoderStack decoder{nullptr};         // the deformable decoder layers + LQE
};
TORCH_MODULE(DFINETransformer);

}  // namespace detr::models
