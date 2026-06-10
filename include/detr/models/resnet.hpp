// Copyright 2026 detrcpp authors. Apache-2.0.
//
// torchvision-style ResNet backbone (FrozenBN-compatible naming: conv1/bn1/
// layer1..4). |bottleneck| picks the block: BasicBlock (R18/R34) or Bottleneck
// (R50/R101+). Block counts pick the depth: {2,2,2,2}=R18, {3,4,6,3}=R34/R50,
// {3,4,23,3}=R101. |dc5| dilates C5 (output stride 16). Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include <array>
#include <vector>

#include "detr/models/frozen_batchnorm.hpp"

namespace detr::models {

struct ResNetImpl : torch::nn::Module {
  torch::nn::Conv2d conv1{nullptr};
  FrozenBatchNorm2d bn1{nullptr};
  torch::nn::Sequential layer1{nullptr};
  torch::nn::Sequential layer2{nullptr};
  torch::nn::Sequential layer3{nullptr};
  torch::nn::Sequential layer4{nullptr};
  bool bottleneck_;

  ResNetImpl(const std::vector<int>& blocks, bool bottleneck, bool dc5);

  torch::Tensor forward(torch::Tensor x);  // C5 only
  // {C3, C4, C5} = layer2/layer3/layer4 outputs.
  std::vector<torch::Tensor> forward_features(torch::Tensor x);
  // Channels of {C3, C4, C5}: {128,256,512} (basic) or {512,1024,2048} (bottleneck).
  std::array<int, 3> feature_channels() const;
};
TORCH_MODULE(ResNet);

}  // namespace detr::models
