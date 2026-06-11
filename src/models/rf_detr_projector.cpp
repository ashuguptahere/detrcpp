// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/rf_detr_projector.hpp"

namespace detr::models {

ChannelLayerNormImpl::ChannelLayerNormImpl(int channels) {
  weight = register_parameter("weight", torch::ones({channels}));
  bias = register_parameter("bias", torch::zeros({channels}));
}

torch::Tensor ChannelLayerNormImpl::forward(torch::Tensor x) {
  // Normalize over the channel dim (dim 1) of [N,C,H,W], like ConvNeXt's LayerNorm.
  auto u = x.mean(1, /*keepdim=*/true);
  auto s = (x - u).pow(2).mean(1, /*keepdim=*/true);
  x = (x - u) / torch::sqrt(s + eps_);
  return weight.view({1, -1, 1, 1}) * x + bias.view({1, -1, 1, 1});
}

ConvXImpl::ConvXImpl(int in_ch, int out_ch, int kernel, int stride) {
  conv = register_module(
      "conv", nn::Conv2d(nn::Conv2dOptions(in_ch, out_ch, kernel).stride(stride).padding(kernel / 2).bias(false)));
  bn = register_module("bn", ChannelLayerNorm(out_ch));
}

torch::Tensor ConvXImpl::forward(torch::Tensor x) {
  return torch::silu(bn->forward(conv->forward(x.contiguous())));
}

namespace {
// A C2f bottleneck: two 3x3 ConvX, no residual (C2f's default shortcut=false).
struct BottleneckImpl : nn::Module {
  BottleneckImpl(int c) {
    cv1 = register_module("cv1", ConvX(c, c, 3, 1));
    cv2 = register_module("cv2", ConvX(c, c, 3, 1));
  }
  torch::Tensor forward(torch::Tensor x) { return cv2->forward(cv1->forward(x)); }
  ConvX cv1{nullptr}, cv2{nullptr};
};
TORCH_MODULE(Bottleneck);
}  // namespace

C2fImpl::C2fImpl(int c1, int c2, int n) : c_(c2 / 2) {
  cv1 = register_module("cv1", ConvX(c1, 2 * c_, 1, 1));
  cv2 = register_module("cv2", ConvX((2 + n) * c_, c2, 1, 1));
  m = register_module("m", nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    m->push_back(Bottleneck(c_));
  }
}

torch::Tensor C2fImpl::forward(torch::Tensor x) {
  auto y0 = cv1->forward(x);
  std::vector<torch::Tensor> ys = y0.split(c_, 1);  // [first c_, second c_]
  for (const auto& mod : *m) {
    ys.push_back(mod->as<BottleneckImpl>()->forward(ys.back()));
  }
  return cv2->forward(torch::cat(ys, 1));
}

RfDetrProjectorImpl::RfDetrProjectorImpl(int num_features, int in_ch, int out_ch, int num_blocks) {
  stage = register_module("stage", C2f(num_features * in_ch, out_ch, num_blocks));
  norm = register_module("norm", ChannelLayerNorm(out_ch));
}

torch::Tensor RfDetrProjectorImpl::forward(const std::vector<torch::Tensor>& feats) {
  // Single-scale: identity sampling -> concat over channels -> C2f -> channel LN.
  auto fused = feats.size() == 1 ? feats[0] : torch::cat(feats, 1);
  return norm->forward(stage->forward(fused));
}

}  // namespace detr::models
