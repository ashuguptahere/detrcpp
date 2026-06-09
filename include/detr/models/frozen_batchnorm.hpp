// Copyright 2026 detrcpp authors. Apache-2.0.
//
// FrozenBatchNorm2d: BatchNorm with affine parameters AND running statistics
// fixed — all four tensors are buffers, never updated and never given a
// gradient. This is what the DETR family uses in the ResNet backbone. At
// inference it is numerically identical to a standard BatchNorm2d in eval mode,
// but during training it does NOT drift the running stats (unlike a plain
// BatchNorm left in train mode), which is what the reference recipe requires.
// eps matches torch's BatchNorm2d default (1e-5). Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace detr::models {

struct FrozenBatchNorm2dImpl : torch::nn::Module {
  torch::Tensor weight{nullptr};
  torch::Tensor bias{nullptr};
  torch::Tensor running_mean{nullptr};
  torch::Tensor running_var{nullptr};
  double eps_;

  explicit FrozenBatchNorm2dImpl(std::int64_t num_features, double eps = 1e-5) : eps_(eps) {
    // Buffers (not parameters): loaded by name like BatchNorm2d's, but excluded
    // from the optimizer and from autograd. num_batches_tracked is intentionally
    // absent (unused for frozen stats).
    weight = register_buffer("weight", torch::ones({num_features}));
    bias = register_buffer("bias", torch::zeros({num_features}));
    running_mean = register_buffer("running_mean", torch::zeros({num_features}));
    running_var = register_buffer("running_var", torch::ones({num_features}));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // Fold into a single per-channel scale/shift; identical to BatchNorm2d's
    // eval-mode output: (x - mean) / sqrt(var + eps) * weight + bias.
    auto scale = weight * (running_var + eps_).rsqrt();
    auto shift = bias - running_mean * scale;
    return x * scale.view({1, -1, 1, 1}) + shift.view({1, -1, 1, 1});
  }
};
TORCH_MODULE(FrozenBatchNorm2d);

}  // namespace detr::models
