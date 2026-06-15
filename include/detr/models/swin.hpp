// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Swin Transformer backbone (the mmdet / Swin-Transformer-Object-Detection layout
// used by Lite-DETR and Salience-DETR). A hierarchical vision transformer: a 4x4
// patch embed, four stages of windowed self-attention (alternating W-MSA / shifted
// SW-MSA) with relative-position bias, and PatchMerging downsampling between stages.
// Each output stage carries its own LayerNorm (`norm0`..`norm3`). forward(image)
// returns the requested stage feature maps as [B, C, H, W] at strides 4/8/16/32.
// Parameterised for Swin-T/S/B/L. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

namespace detr::models {

namespace nn = torch::nn;

// Image -> patch embedding: a patch_size x patch_size strided conv, then (optionally)
// a LayerNorm over the flattened tokens. Returns the token map as [B, embed_dim, H, W].
struct SwinPatchEmbedImpl : nn::Module {
  SwinPatchEmbedImpl(int patch_size, int in_chans, int embed_dim, bool with_norm);
  torch::Tensor forward(torch::Tensor x);
  int patch_size_;
  int embed_dim_;
  nn::Conv2d proj{nullptr};
  nn::LayerNorm norm{nullptr};  // null when norm == false
};
TORCH_MODULE(SwinPatchEmbed);

// Window-based multi-head self-attention with a learned relative-position bias.
// `relative_position_index` is a (persistent) buffer — it ships in the upstream
// checkpoint and moves with the module across devices.
struct SwinWindowAttentionImpl : nn::Module {
  SwinWindowAttentionImpl(int dim, int window_size, int num_heads);
  // x: [num_windows*B, N, C]; mask: [num_windows, N, N] or undefined.
  torch::Tensor forward(torch::Tensor x, const torch::Tensor& mask);
  int window_size_;
  int num_heads_;
  double scale_;
  torch::Tensor relative_position_bias_table;  // [(2W-1)^2, num_heads]
  torch::Tensor relative_position_index;        // [N, N] long buffer
  nn::Linear qkv{nullptr}, proj{nullptr};
};
TORCH_MODULE(SwinWindowAttention);

// One Swin block: pre-norm windowed attention (with optional cyclic shift) + residual,
// then pre-norm GELU MLP + residual. Padding to a window multiple is handled internally.
struct SwinBlockImpl : nn::Module {
  SwinBlockImpl(int dim, int num_heads, int window_size, int shift_size, double mlp_ratio);
  // x: [B, H*W, C]; mask_matrix: SW-MSA attention mask for this stage (or undefined).
  torch::Tensor forward(torch::Tensor x, int H, int W, const torch::Tensor& mask_matrix);
  int window_size_, shift_size_;
  nn::LayerNorm norm1{nullptr}, norm2{nullptr};
  SwinWindowAttention attn{nullptr};
  nn::Linear fc1{nullptr}, fc2{nullptr};  // registered under "mlp"
};
TORCH_MODULE(SwinBlock);

// Patch merging: concatenate the four strided 2x2 sub-samples (4*C), LayerNorm, then a
// bias-free Linear back to 2*C. Halves spatial resolution, doubles channels.
struct SwinPatchMergingImpl : nn::Module {
  explicit SwinPatchMergingImpl(int dim);
  torch::Tensor forward(torch::Tensor x, int H, int W);
  nn::Linear reduction{nullptr};
  nn::LayerNorm norm{nullptr};
};
TORCH_MODULE(SwinPatchMerging);

// Result of one stage: the stage features at the input resolution (`out`, pre per-stage
// norm), and the downsampled features feeding the next stage (`down`, at Wh x Ww).
struct SwinStageOut {
  torch::Tensor out;
  torch::Tensor down;
  int Wh;
  int Ww;
};

// One Swin stage: `depth` blocks (even = W-MSA, odd = SW-MSA) + an optional downsample.
struct SwinBasicLayerImpl : nn::Module {
  SwinBasicLayerImpl(int dim, int depth, int num_heads, int window_size, double mlp_ratio,
                     bool with_downsample);
  SwinStageOut forward(torch::Tensor x, int H, int W);
  int window_size_, shift_size_;
  nn::ModuleList blocks{nullptr};
  SwinPatchMerging downsample{nullptr};  // null when this stage does not downsample
};
TORCH_MODULE(SwinBasicLayer);

// Swin variant geometry. T/S/B/L differ only in embed_dim / depths / num_heads /
// window_size. `dilation` keeps stage 3 at the stride-16 resolution (no downsample
// after stage 2) for a 16x output — the Lite-DETR high-resolution configuration.
struct SwinConfig {
  int embed_dim = 96;
  std::vector<int> depths{2, 2, 6, 2};
  std::vector<int> num_heads{3, 6, 12, 24};
  int window_size = 7;
  int patch_size = 4;
  double mlp_ratio = 4.0;
  std::vector<int> out_indices{0, 1, 2, 3};
  bool dilation = false;
};

// Swin Transformer backbone. forward(image) returns the `out_indices` stage feature
// maps as [B, C, H, W]; out_channels() reports their channel counts.
struct SwinTransformerImpl : nn::Module {
  explicit SwinTransformerImpl(const SwinConfig& cfg);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  std::vector<int> out_channels() const;

  std::vector<int> out_indices_;
  std::vector<int> num_features_;
  SwinPatchEmbed patch_embed{nullptr};
  nn::ModuleList layers{nullptr};
  std::vector<nn::LayerNorm> out_norms_;  // one per out_index, registered "norm{i}"
};
TORCH_MODULE(SwinTransformer);

// The five canonical Swin sizes (matching build_swin_transformer in the DINO / Lite-DETR
// reference). `out_indices` / `dilation` are left at the SwinConfig defaults.
SwinConfig SwinConfigFor(const std::string& name);

}  // namespace detr::models
