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

ConvXImpl::ConvXImpl(int in_ch, int out_ch, int kernel, int stride, bool batch_norm, ConvAct act)
    : act_(act) {
  conv = register_module(
      "conv", nn::Conv2d(nn::Conv2dOptions(in_ch, out_ch, kernel).stride(stride).padding(kernel / 2).bias(false)));
  if (batch_norm) {
    bn2d = register_module("bn", nn::BatchNorm2d(nn::BatchNorm2dOptions(out_ch).eps(1e-5)));
  } else {
    bn = register_module("bn", ChannelLayerNorm(out_ch));
  }
}

torch::Tensor ConvXImpl::forward(torch::Tensor x) {
  auto c = conv->forward(x.contiguous());
  auto n = bn2d ? bn2d->forward(c) : bn->forward(c);
  switch (act_) {
    case ConvAct::kSiLU:
      return torch::silu(n);
    case ConvAct::kReLU:
      return torch::relu(n);
    case ConvAct::kNone:
      return n;
  }
  return n;
}

namespace {
// A C2f bottleneck: two 3x3 ConvX, no residual (C2f's default shortcut=false).
struct BottleneckImpl : nn::Module {
  BottleneckImpl(int c, bool batch_norm) {
    cv1 = register_module("cv1", ConvX(c, c, 3, 1, batch_norm));
    cv2 = register_module("cv2", ConvX(c, c, 3, 1, batch_norm));
  }
  torch::Tensor forward(torch::Tensor x) { return cv2->forward(cv1->forward(x)); }
  ConvX cv1{nullptr}, cv2{nullptr};
};
TORCH_MODULE(Bottleneck);
}  // namespace

C2fImpl::C2fImpl(int c1, int c2, int n, bool batch_norm) : c_(c2 / 2) {
  cv1 = register_module("cv1", ConvX(c1, 2 * c_, 1, 1, batch_norm));
  cv2 = register_module("cv2", ConvX((2 + n) * c_, c2, 1, 1, batch_norm));
  m = register_module("m", nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    m->push_back(Bottleneck(c_, batch_norm));
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

RfDetrProjectorImpl::RfDetrProjectorImpl(int num_features, int in_ch, int out_ch, int num_blocks,
                                         bool batch_norm) {
  stage = register_module("stage", C2f(num_features * in_ch, out_ch, num_blocks, batch_norm));
  norm = register_module("norm", ChannelLayerNorm(out_ch));
}

torch::Tensor RfDetrProjectorImpl::forward(const std::vector<torch::Tensor>& feats) {
  // Single-scale: identity sampling -> concat over channels -> C2f -> channel LN.
  auto fused = feats.size() == 1 ? feats[0] : torch::cat(feats, 1);
  return norm->forward(stage->forward(fused));
}

namespace {

// Resamples one backbone feature to a target scale: 2.0 upsamples (ConvTranspose,
// preceded by a 1x1 ConvNorm when the input is wide >512); 0.5 downsamples (stride-2
// ConvNorm). All convs use ReLU + BatchNorm. Layers register as "layers.{i}".
struct SamplingLayerImpl : nn::Module {
  SamplingLayerImpl(int in_ch, double scale) {
    layers = register_module("layers", nn::ModuleList());
    if (scale == 2.0) {
      if (in_ch > 512) {
        layers->push_back(ConvX(in_ch, in_ch / 2, 1, 1, true, ConvAct::kReLU));
        layers->push_back(nn::ConvTranspose2d(
            nn::ConvTranspose2dOptions(in_ch / 2, in_ch / 4, 2).stride(2)));
      } else {
        layers->push_back(
            nn::ConvTranspose2d(nn::ConvTranspose2dOptions(in_ch, in_ch / 2, 2).stride(2)));
      }
    } else {  // 0.5 downsample
      layers->push_back(ConvX(in_ch, in_ch, 3, 2, true, ConvAct::kReLU));
    }
  }
  torch::Tensor forward(torch::Tensor x) {
    for (const auto& m : *layers) {
      if (auto* c = m->as<ConvXImpl>()) {
        x = c->forward(x);
      } else {
        x = m->as<nn::ConvTranspose2dImpl>()->forward(x);
      }
    }
    return x;
  }
  nn::ModuleList layers{nullptr};
};
TORCH_MODULE(SamplingLayer);

// One output scale: resample each feature, concat, C2f -> channel LN.
struct ScaleProjectorImpl : nn::Module {
  ScaleProjectorImpl(int num_features, int in_ch, int out_ch, int num_blocks, double scale) {
    int interm = in_ch;
    if (scale == 2.0) {
      interm = in_ch > 512 ? in_ch / 4 : in_ch / 2;
    }
    sampling = register_module("sampling", nn::ModuleList());
    for (int f = 0; f < num_features; ++f) {
      sampling->push_back(SamplingLayer(in_ch, scale));
    }
    stage = register_module("stage", C2f(interm * num_features, out_ch, num_blocks, /*bn=*/true));
    norm = register_module("norm", ChannelLayerNorm(out_ch));
  }
  torch::Tensor forward(const std::vector<torch::Tensor>& feats) {
    std::vector<torch::Tensor> sampled;
    sampled.reserve(feats.size());
    for (std::size_t f = 0; f < feats.size(); ++f) {
      sampled.push_back(sampling[f]->as<SamplingLayerImpl>()->forward(feats[f]));
    }
    return norm->forward(stage->forward(torch::cat(sampled, 1)));
  }
  nn::ModuleList sampling{nullptr};
  C2f stage{nullptr};
  ChannelLayerNorm norm{nullptr};
};
TORCH_MODULE(ScaleProjector);

}  // namespace

LwDetrMultiScaleProjectorImpl::LwDetrMultiScaleProjectorImpl(int num_features, int in_ch, int out_ch,
                                                             int num_blocks,
                                                             std::vector<double> scale_factors)
    : scales_(std::move(scale_factors)) {
  scale_layers = register_module("scale_layers", nn::ModuleList());
  for (double s : scales_) {
    scale_layers->push_back(ScaleProjector(num_features, in_ch, out_ch, num_blocks, s));
  }
}

std::vector<torch::Tensor> LwDetrMultiScaleProjectorImpl::forward(
    const std::vector<torch::Tensor>& feats) {
  std::vector<torch::Tensor> outs;
  outs.reserve(scales_.size());
  for (const auto& m : *scale_layers) {
    outs.push_back(m->as<ScaleProjectorImpl>()->forward(feats));
  }
  return outs;
}

}  // namespace detr::models
