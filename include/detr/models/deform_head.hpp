// Copyright 2026 detrcpp authors. Apache-2.0.
//
// The shared deformable detection head used by RT-DETR, RF-DETR, and DINO: given
// a flattened multi-scale memory, it runs IoU-aware query selection over grid
// anchors and a deformable decoder (MSDeformAttn, 4D reference points) with
// per-layer iterative box refinement, producing the focal class logits + boxes.
// Models differ only in how they build the memory. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include "detr/models/deform_attn.hpp"  // SpatialShapes
#include "detr/models/model.hpp"        // Detections

namespace detr::models {

// n-layer MLP with ReLU between (shared by the detection heads).
struct MlpImpl : torch::nn::Module {
  torch::nn::ModuleList layers{nullptr};
  int n_;
  MlpImpl(int in, int hidden, int out, int n);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Mlp);

// Handles to the head submodules (owned by the model that registered them).
struct DeformDetectHead {
  torch::nn::Linear enc_output{nullptr};
  torch::nn::LayerNorm enc_output_norm{nullptr};
  torch::nn::Linear enc_score_head{nullptr};
  Mlp enc_bbox_head{nullptr};
  Mlp query_pos_head{nullptr};
  torch::nn::ModuleList decoder{nullptr};
  torch::nn::ModuleList dec_score{nullptr};
  torch::nn::ModuleList dec_bbox{nullptr};
  int num_levels{0};
  int num_queries{0};
};

DeformDetectHead BuildDeformDetectHead(torch::nn::Module& model, int d, int levels, int heads,
                                       int points, int ff, int dec_layers, int num_classes,
                                       int num_queries, bool discrete_sample = false);

// Optional contrastive-denoising prefix (DINO-CDN). When active, num_dn denoising
// queries are PREPENDED to the topk matching queries and the joint decoder runs
// under attn_mask. Empty/inactive by default.
struct DeformCdn {
  bool active{false};
  torch::Tensor dn_tgt;     // [B, num_dn, d] content (label-embedded by the model)
  torch::Tensor dn_ref;     // [B, num_dn, 4] noised anchor in sigmoid space (0,1)
  torch::Tensor attn_mask;  // [L, L] bool, true = BLOCK (L = num_dn + num_queries)
  int num_dn{0};
};

// memory: [B, Sum(H*W), d]. Returns focal logits [B, nq, num_classes] + boxes
// [B, nq, 4] (cxcywh).
Detections RunDeformDetectHead(const DeformDetectHead& head, torch::Tensor memory,
                               const SpatialShapes& shapes);

// CDN variant: prepends cdn.dn_* under cdn.attn_mask, runs the joint decoder, and
// splits per-layer outputs — the matching slice is returned, the denoising slice
// fills dn_out. cdn.active==false (or eval) => identical to the 3-arg overload.
Detections RunDeformDetectHead(const DeformDetectHead& head, torch::Tensor memory,
                               const SpatialShapes& shapes, const DeformCdn& cdn,
                               DenoisingOut& dn_out);

}  // namespace detr::models
