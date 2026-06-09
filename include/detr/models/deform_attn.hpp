// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Multi-scale deformable attention (Deformable-DETR's MSDeformAttn) — the shared
// building block for the deformable DETR family (Deformable-DETR, DINO, RT-DETR).
// The core sampling uses grid_sample (also an ONNX op), so it stays export-able.
// Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace detr::models {

using SpatialShapes = std::vector<std::pair<std::int64_t, std::int64_t>>;

// The learnable-parameter-free core: bilinearly samples |value| at
// |sampling_locations| (normalized [0,1]) per level and weights by
// |attention_weights|. Mirrors Deformable-DETR's ms_deform_attn_core_pytorch.
//   value:              [N, Sum(H*W), n_heads, head_dim]
//   sampling_locations: [N, Lq, n_heads, n_levels, n_points, 2]
//   attention_weights:  [N, Lq, n_heads, n_levels, n_points]
// returns:              [N, Lq, n_heads*head_dim]
torch::Tensor MSDeformAttnCore(const torch::Tensor& value, const SpatialShapes& shapes,
                               const torch::Tensor& sampling_locations,
                               const torch::Tensor& attention_weights);

// Full MSDeformAttn module: projects value, predicts sampling offsets +
// attention weights from the query, calls the core, and output-projects.
struct MSDeformAttnImpl : torch::nn::Module {
  int d_model_;
  int n_levels_;
  int n_heads_;
  int n_points_;
  torch::nn::Linear sampling_offsets{nullptr};
  torch::nn::Linear attention_weights{nullptr};
  torch::nn::Linear value_proj{nullptr};
  torch::nn::Linear output_proj{nullptr};

  MSDeformAttnImpl(int d_model, int n_levels, int n_heads, int n_points);

  // query:            [N, Lq, d_model]
  // reference_points: [N, Lq, n_levels, 2]  (normalized centers per level)
  // input_flatten:    [N, Sum(H*W), d_model]
  torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& reference_points,
                        const torch::Tensor& input_flatten, const SpatialShapes& shapes);
};
TORCH_MODULE(MSDeformAttn);

}  // namespace detr::models
