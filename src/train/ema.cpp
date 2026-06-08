// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/ema.hpp"

#include <cstddef>

#include "detr/weights/torch_bridge.hpp"

namespace detr::train {

ModelEma::ModelEma(const torch::nn::Module& model, double decay) : decay_(decay) {
  torch::NoGradGuard no_grad;
  for (const auto& p : model.named_parameters(/*recurse=*/true)) {
    names_.push_back(p.key());
    shadow_.push_back(p.value().detach().clone());
  }
}

void ModelEma::Update(const torch::nn::Module& model) {
  torch::NoGradGuard no_grad;
  std::size_t i = 0;
  for (const auto& p : model.named_parameters(/*recurse=*/true)) {
    // shadow = decay * shadow + (1 - decay) * param
    shadow_[i].mul_(decay_).add_(p.value().detach(), 1.0 - decay_);
    ++i;
  }
}

void ModelEma::CopyTo(torch::nn::Module& model) const {
  torch::NoGradGuard no_grad;
  std::size_t i = 0;
  for (const auto& p : model.named_parameters(/*recurse=*/true)) {
    p.value().copy_(shadow_[i]);
    ++i;
  }
}

weights::StateDict ModelEma::State() const {
  weights::StateDict sd;
  for (std::size_t i = 0; i < names_.size(); ++i) {
    auto raw = weights::FromTensor(shadow_[i]);
    if (raw) {
      sd.Set(names_[i], std::move(*raw));
    }
  }
  return sd;
}

void ModelEma::LoadState(const weights::StateDict& state) {
  torch::NoGradGuard no_grad;
  for (std::size_t i = 0; i < names_.size(); ++i) {
    const auto* raw = state.Find(names_[i]);
    if (raw != nullptr) {
      shadow_[i].copy_(weights::ToTensor(*raw));
    }
  }
}

}  // namespace detr::train
