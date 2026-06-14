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

// Conv activation: SiLU (C2f convs), ReLU (LW-DETR sampling convs), or none.
enum class ConvAct { kSiLU, kReLU, kNone };

// Conv(no bias) + norm + activation. The conv norm is a channel LayerNorm (RF-DETR)
// or a BatchNorm2d (LW-DETR) — `batch_norm` selects; both register as "bn".
struct ConvXImpl : nn::Module {
  ConvXImpl(int in_ch, int out_ch, int kernel, int stride, bool batch_norm = false,
            ConvAct act = ConvAct::kSiLU);
  torch::Tensor forward(torch::Tensor x);
  ConvAct act_;
  nn::Conv2d conv{nullptr};
  ChannelLayerNorm bn{nullptr};   // RF-DETR
  nn::BatchNorm2d bn2d{nullptr};  // LW-DETR
};
TORCH_MODULE(ConvX);

// C2f (YOLOv8): cv1 splits into two, n bottlenecks chain off the second half,
// everything concatenated and fused by cv2. Bottlenecks here have no shortcut.
struct C2fImpl : nn::Module {
  C2fImpl(int c1, int c2, int n, bool batch_norm = false);
  torch::Tensor forward(torch::Tensor x);
  int c_;
  ConvX cv1{nullptr}, cv2{nullptr};
  nn::ModuleList m{nullptr};  // each entry is a Bottleneck (cv1+cv2 ConvX, no residual)
};
TORCH_MODULE(C2f);

// Single-scale projector: concat |num_features| backbone features (each in_ch) ->
// C2f(sum, out_ch, num_blocks) -> channel LayerNorm. Returns one [B, out_ch, H, W].
// `batch_norm` switches the C2f conv norm (RF-DETR LayerNorm vs LW-DETR BatchNorm).
struct RfDetrProjectorImpl : nn::Module {
  RfDetrProjectorImpl(int num_features, int in_ch, int out_ch, int num_blocks,
                      bool batch_norm = false);
  torch::Tensor forward(const std::vector<torch::Tensor>& feats);
  C2f stage{nullptr};
  ChannelLayerNorm norm{nullptr};
};
TORCH_MODULE(RfDetrProjector);

// LW-DETR multi-scale projector (LwDetrMultiScaleProjector): for each output scale
// (e.g. P3 via 2x upsample, P5 via 0.5x downsample), each of |num_features| backbone
// features (each in_ch) is resampled to that scale, concatenated, run through a C2f
// (BatchNorm) + channel LayerNorm. Returns one [B, out_ch, H_s, W_s] per scale.
struct LwDetrMultiScaleProjectorImpl : nn::Module {
  LwDetrMultiScaleProjectorImpl(int num_features, int in_ch, int out_ch, int num_blocks,
                                std::vector<double> scale_factors);
  std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& feats);
  std::vector<double> scales_;
  nn::ModuleList scale_layers{nullptr};  // one ScaleProjector per scale
};
TORCH_MODULE(LwDetrMultiScaleProjector);

}  // namespace detr::models
