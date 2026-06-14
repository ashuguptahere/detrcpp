// Copyright 2026 detrcpp authors. Apache-2.0.
//
// HGNetv2 backbone (PaddleDetection PPHGNetV2, as used by D-FINE). A stride-4 stem
// then four HG stages; each stage optionally downsamples (depthwise 3x3 stride-2)
// and stacks HG blocks. An HG block runs `layer_num` convs, concatenates the input
// with every intermediate, and fuses them with a squeeze->excite 1x1 pair ("se"
// aggregation). Optional LearnableAffineBlock (per-channel-free scalar scale/shift)
// follows each activated conv when `use_lab` (D-FINE n/s/m). BatchNorm is frozen
// (eval-identical), matching the other detrcpp backbones. Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "detr/models/frozen_batchnorm.hpp"

namespace detr::models {

namespace nn = torch::nn;

// scale * x + bias, with scalar (shape [1]) learnable scale and bias.
struct HgLabImpl : nn::Module {
  HgLabImpl();
  torch::Tensor forward(const torch::Tensor& x);
  torch::Tensor scale, bias;
};
TORCH_MODULE(HgLab);

// Conv(no bias) -> FrozenBN -> ReLU? -> LAB?. The LAB is present only when the conv
// is activated AND use_lab (matches the upstream `use_act and use_lab`).
struct HgConvImpl : nn::Module {
  HgConvImpl(int in_ch, int out_ch, int kernel, int stride, int groups, bool use_act, bool use_lab);
  torch::Tensor forward(torch::Tensor x);
  bool use_act_;
  nn::Conv2d conv{nullptr};
  FrozenBatchNorm2d bn{nullptr};
  HgLab lab{nullptr};  // null unless use_act && use_lab
};
TORCH_MODULE(HgConv);

// Depthwise-separable conv pair: 1x1 (no act) then kxk depthwise (act). Used by the
// "light" HG blocks (stages 3/4).
struct HgLightConvImpl : nn::Module {
  HgLightConvImpl(int in_ch, int out_ch, int kernel, bool use_lab);
  torch::Tensor forward(torch::Tensor x);
  HgConv conv1{nullptr}, conv2{nullptr};
};
TORCH_MODULE(HgLightConv);

// Stride-4 stem: stem1 (3x3 s2), a two-branch 2x2 path (stem2a/stem2b) fused with a
// stride-1 maxpool, then stem3 (3x3 s2) + stem4 (1x1).
struct HgStemImpl : nn::Module {
  HgStemImpl(int in_ch, int mid_ch, int out_ch, bool use_lab);
  torch::Tensor forward(torch::Tensor x);
  HgConv stem1{nullptr}, stem2a{nullptr}, stem2b{nullptr}, stem3{nullptr}, stem4{nullptr};
  nn::MaxPool2d pool{nullptr};
};
TORCH_MODULE(HgStem);

// One HG block: `layer_num` (light or plain) convs, concat [input, all intermediates],
// squeeze->excite 1x1 aggregation, optional residual add (when in==out).
struct HgBlockImpl : nn::Module {
  HgBlockImpl(int in_ch, int mid_ch, int out_ch, int layer_num, int kernel, bool residual,
              bool light, bool use_lab);
  torch::Tensor forward(torch::Tensor x);
  bool residual_, light_;
  nn::ModuleList layers{nullptr};       // HgConv or HgLightConv
  nn::ModuleList aggregation{nullptr};  // [squeeze HgConv, excite HgConv]
};
TORCH_MODULE(HgBlock);

// One HG stage: optional depthwise stride-2 downsample, then `block_num` HG blocks.
struct HgStageImpl : nn::Module {
  HgStageImpl(int in_ch, int mid_ch, int out_ch, int block_num, int layer_num, bool downsample,
              bool light, int kernel, bool use_lab);
  torch::Tensor forward(torch::Tensor x);
  HgConv downsample{nullptr};  // null when the stage does not downsample
  nn::ModuleList blocks{nullptr};
};
TORCH_MODULE(HgStage);

// HGNetv2 backbone. `variant` is one of "B0"/"B2"/"B4"/"B5" (D-FINE n,s / m / l / x);
// returns the feature maps at `return_idx` (e.g. {1,2,3} = strides 8/16/32).
struct HgNetV2Impl : nn::Module {
  HgNetV2Impl(const std::string& variant, bool use_lab, std::vector<int> return_idx);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  // Output channels of the returned levels (for wiring the neck).
  std::vector<int> out_channels() const { return out_channels_; }

  std::vector<int> return_idx_;
  std::vector<int> out_channels_;
  HgStem stem{nullptr};
  nn::ModuleList stages{nullptr};
};
TORCH_MODULE(HgNetV2);

}  // namespace detr::models
