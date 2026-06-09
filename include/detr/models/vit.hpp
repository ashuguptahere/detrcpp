// Copyright 2026 detrcpp authors. Apache-2.0.
//
// A plain Vision Transformer backbone (DINOv2-style structure: Conv patch embed,
// 2D sine position, pre-norm transformer blocks). Returns a single feature map
// [B, embed_dim, H/patch, W/patch]; RF-DETR projects it to multiple scales for
// the deformable decoder. (Exact DINOv2 windowed attention / register tokens are
// a tracked follow-up.) Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

namespace detr::models {

struct ViTImpl : torch::nn::Module {
  torch::nn::Conv2d patch_embed{nullptr};
  torch::nn::ModuleList blocks{nullptr};
  torch::nn::LayerNorm norm{nullptr};
  int embed_dim_;
  int patch_;
  int nheads_;

  ViTImpl(int embed_dim, int depth, int nheads, int patch, int mlp_ratio);

  torch::Tensor forward(torch::Tensor x);  // [B,3,H,W] -> [B, embed_dim, H/patch, W/patch]
  int embed_dim() const { return embed_dim_; }
};
TORCH_MODULE(ViT);

}  // namespace detr::models
