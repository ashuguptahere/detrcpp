// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RF-DETR's multi-scale projector (roboflow/rf-detr MultiScaleProjector): fuses the
// DINOv2 backbone's out_indices features into the deformable decoder's feature
// level(s) via a YOLOv8-style C2f neck. The nano/single-scale path concatenates the
// (identity-sampled) features and runs one C2f stage + a channel LayerNorm.
// Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include <vector>

namespace detr::models {

namespace nn = torch::nn;

// ConvNeXt-style channel-first LayerNorm over the channel dim of an NCHW tensor.
struct ChannelLayerNormImpl : nn::Module {
  explicit ChannelLayerNormImpl(int channels);
  torch::Tensor forward(torch::Tensor x);
  torch::Tensor weight, bias;
  double eps_{1e-6};
};
TORCH_MODULE(ChannelLayerNorm);

// Conv(no bias) + channel LayerNorm + SiLU.
struct ConvXImpl : nn::Module {
  ConvXImpl(int in_ch, int out_ch, int kernel, int stride);
  torch::Tensor forward(torch::Tensor x);
  nn::Conv2d conv{nullptr};
  ChannelLayerNorm bn{nullptr};
};
TORCH_MODULE(ConvX);

// C2f (YOLOv8): cv1 splits into two, n bottlenecks chain off the second half,
// everything concatenated and fused by cv2. Bottlenecks here have no shortcut.
struct C2fImpl : nn::Module {
  C2fImpl(int c1, int c2, int n);
  torch::Tensor forward(torch::Tensor x);
  int c_;
  ConvX cv1{nullptr}, cv2{nullptr};
  nn::ModuleList m{nullptr};  // each entry is a Bottleneck (cv1+cv2 ConvX, no residual)
};
TORCH_MODULE(C2f);

// Single-scale projector: concat |num_features| backbone features (each in_ch) ->
// C2f(sum, out_ch, num_blocks) -> channel LayerNorm. Returns one [B, out_ch, H, W].
struct RfDetrProjectorImpl : nn::Module {
  RfDetrProjectorImpl(int num_features, int in_ch, int out_ch, int num_blocks);
  torch::Tensor forward(const std::vector<torch::Tensor>& feats);
  C2f stage{nullptr};
  ChannelLayerNorm norm{nullptr};
};
TORCH_MODULE(RfDetrProjector);

}  // namespace detr::models
