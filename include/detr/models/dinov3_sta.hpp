// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DINOv3-STA backbone (DEIMv2 s/m/l/x). A plain RoPE Vision Transformer (the DEIMv2
// authors' distilled ViT-Tiny for s/m; Meta's DINOv3 ViT for l/x — the latter is a
// follow-up) feeds a Spatial-Tuning Adapter (STA): the ViT's intermediate patch-token
// maps are reshaped to a 1/16 grid and resampled to 1/8, 1/16, 1/32, while a lite
// SpatialPriorModule runs a small conv stem on the raw image to recover the same three
// scales; the two are concatenated and fused by a 1x1 conv + BatchNorm per level.
// Output: three `hidden_dim` feature maps at strides 8/16/32. Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

namespace detr::models {

namespace nn = torch::nn;

// 2D rotary position embedding (a `periods` buffer; sin/cos for a HxW patch grid).
struct RopeEmbedImpl : nn::Module {
  explicit RopeEmbedImpl(int head_dim);
  // Returns {sin, cos}, each [1, 1, H*W, head_dim], for the patch tokens.
  std::pair<torch::Tensor, torch::Tensor> SinCos(int h, int w) const;
  torch::Tensor periods;  // [head_dim / 4]
};
TORCH_MODULE(RopeEmbed);

// One pre-norm ViT block with RoPE self-attention (cls token is left un-rotated).
struct VitRopeBlockImpl : nn::Module {
  VitRopeBlockImpl(int dim, int num_heads, double mlp_ratio);
  torch::Tensor forward(torch::Tensor x, const torch::Tensor& sin, const torch::Tensor& cos);
  int num_heads_;
  double scale_;
  nn::LayerNorm norm1{nullptr}, norm2{nullptr};
  nn::Linear qkv{nullptr}, proj{nullptr};
  nn::Linear fc1{nullptr}, fc2{nullptr};
};
TORCH_MODULE(VitRopeBlock);

// The ViT-Tiny model body (matches the upstream `_model` submodule naming).
struct VitRopeModelImpl : nn::Module {
  VitRopeModelImpl(int embed_dim, int num_heads, int depth, int patch_size, double mlp_ratio);
  torch::Tensor cls_token;
  nn::Conv2d patch_embed_proj{nullptr};  // registered under "patch_embed.proj"
  nn::ModuleList blocks{nullptr};
  RopeEmbed rope_embed{nullptr};
};
TORCH_MODULE(VitRopeModel);

// RoPE ViT-Tiny backbone. forward(image) returns the patch-token maps [B, N, D] at the
// requested `interaction_indexes` (in block order).
struct VitRopeImpl : nn::Module {
  VitRopeImpl(int embed_dim, int num_heads, int depth, int patch_size,
              std::vector<int> return_layers, double mlp_ratio = 4.0);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  int embed_dim_, patch_size_;
  std::vector<int> return_layers_;
  VitRopeModel model{nullptr};  // registered as "_model"
};
TORCH_MODULE(VitRope);

// LayerScale: a learnable per-channel gain (gamma) applied to a residual branch.
struct DinoLayerScaleImpl : nn::Module {
  explicit DinoLayerScaleImpl(int dim);
  torch::Tensor forward(const torch::Tensor& x) { return x * gamma; }
  torch::Tensor gamma;
};
TORCH_MODULE(DinoLayerScale);

// One Meta-DINOv3 ViT block: pre-norm RoPE attention + LayerScale, pre-norm FFN
// (GELU MLP for ViT-S, SwiGLU for ViT-S+) + LayerScale. `prefix` tokens (cls + storage)
// are left un-rotated.
struct DinoV3BlockImpl : nn::Module {
  DinoV3BlockImpl(int dim, int num_heads, double ffn_ratio, bool swiglu);
  torch::Tensor forward(torch::Tensor x, const torch::Tensor& sin, const torch::Tensor& cos,
                        int prefix);
  int num_heads_;
  double scale_;
  bool swiglu_;
  nn::LayerNorm norm1{nullptr}, norm2{nullptr};
  nn::Linear qkv{nullptr}, proj{nullptr};
  DinoLayerScale ls1{nullptr}, ls2{nullptr};
  nn::Linear fc1{nullptr}, fc2{nullptr};      // GELU MLP (ViT-S)
  nn::Linear w1{nullptr}, w2{nullptr}, w3{nullptr};  // SwiGLU (ViT-S+)
};
TORCH_MODULE(DinoV3Block);

// Meta DINOv3 ViT (ViT-S/16, ViT-S+/16) backbone. cls + `n_storage` register tokens +
// patches; final LayerNorm applied to each returned intermediate. forward returns the
// patch-token maps [B, N, D] at `interaction_indexes`.
struct DinoV3VitImpl : nn::Module {
  DinoV3VitImpl(int embed_dim, int num_heads, int depth, int patch_size,
                std::vector<int> return_layers, double ffn_ratio, bool swiglu, int n_storage);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  int embed_dim_, patch_size_, n_storage_;
  std::vector<int> return_layers_;
  torch::Tensor cls_token, storage_tokens;
  nn::Conv2d patch_embed_proj{nullptr};  // registered under "patch_embed.proj"
  nn::ModuleList blocks{nullptr};
  RopeEmbed rope_embed{nullptr};
  nn::LayerNorm norm{nullptr};  // final norm applied to each intermediate
};
TORCH_MODULE(DinoV3Vit);

// Lite SpatialPriorModule (STA detail branch): a 1/4 conv stem then 1/8, 1/16, 1/32
// conv stages. BatchNorm (eval-identical to the upstream SyncBatchNorm).
struct SpatialPriorImpl : nn::Module {
  explicit SpatialPriorImpl(int inplanes);
  // Returns {c2 (1/8, 2*inplanes), c3 (1/16, 4*inplanes), c4 (1/32, 4*inplanes)}.
  std::vector<torch::Tensor> forward(torch::Tensor x);
  nn::Sequential stem{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
};
TORCH_MODULE(SpatialPrior);

// Backbone selector: the distilled RoPE ViT-Tiny (s/m) or the Meta DINOv3 ViT (l/x).
struct DinoStaConfig {
  int embed_dim, num_heads, depth = 12, patch_size = 16, conv_inplane, hidden_dim;
  std::vector<int> interaction_indexes;
  bool dinov3_vit = false;   // true: Meta DINOv3 ViT (l/x); false: ViT-Tiny (s/m)
  double ffn_ratio = 4.0;    // DINOv3 FFN ratio (l: 4, x: 6)
  bool swiglu = false;       // DINOv3 FFN type (x: SwiGLU, l: GELU MLP)
  int n_storage = 0;         // DINOv3 register tokens (l/x: 4)
};

// DINOv3-STA backbone: a RoPE ViT + the STA adapter -> three fused `hidden_dim` maps.
struct DinoV3StaImpl : nn::Module {
  explicit DinoV3StaImpl(const DinoStaConfig& cfg);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  std::vector<int> out_channels() const {
    return std::vector<int>(3, hidden_dim_);
  }
  int patch_size_, hidden_dim_;
  VitRope dinov3{nullptr};         // ViT-Tiny (s/m); exactly one backbone is built
  DinoV3Vit dinov3_vit{nullptr};   // Meta DINOv3 ViT (l/x)
  SpatialPrior sta{nullptr};
  nn::ModuleList convs{nullptr};  // 1x1 Conv2d (no bias) per level
  nn::ModuleList norms{nullptr};  // BatchNorm2d per level
};
TORCH_MODULE(DinoV3Sta);

}  // namespace detr::models
