// Copyright 2026 detrcpp authors. Apache-2.0.
//
// torchvision-style ResNet backbone (FrozenBN-compatible naming: conv1/bn1/
// layer1..4 with Bottleneck blocks). Shared by the single-scale DETR models and
// the multi-scale deformable family. Block counts pick the depth: {3,4,6,3}=R50,
// {3,4,23,3}=R101. |dc5| dilates C5 (output stride 16). Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <vector>

#include <torch/torch.h>

namespace detr::models {

struct ResNetImpl : torch::nn::Module {
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Sequential layer1{nullptr};
  torch::nn::Sequential layer2{nullptr};
  torch::nn::Sequential layer3{nullptr};
  torch::nn::Sequential layer4{nullptr};

  ResNetImpl(const std::vector<int>& blocks, bool dc5);

  torch::Tensor forward(torch::Tensor x);  // C5 only: [B, 2048, H/{32,16}, W/{32,16}]
  // {C3, C4, C5} = layer2/layer3/layer4 outputs (512/1024/2048 channels) — the
  // last three feature maps the deformable models consume.
  std::vector<torch::Tensor> forward_features(torch::Tensor x);
};
TORCH_MODULE(ResNet);

}  // namespace detr::models
