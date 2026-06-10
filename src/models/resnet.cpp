// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/resnet.hpp"

#include <array>
#include <vector>

namespace detr::models {

namespace {
namespace nn = torch::nn;

// ResNet-50/101 bottleneck (torchvision naming): conv1/bn1/conv2/bn2/conv3/bn3
// and an optional downsample = Sequential(conv, bn). |dilation| on the 3x3 conv.
struct BottleneckImpl : nn::Module {
  nn::Conv2d conv1{nullptr};
  nn::Conv2d conv2{nullptr};
  nn::Conv2d conv3{nullptr};
  FrozenBatchNorm2d bn1{nullptr};
  FrozenBatchNorm2d bn2{nullptr};
  FrozenBatchNorm2d bn3{nullptr};
  nn::Sequential downsample{nullptr};

  BottleneckImpl(int in, int planes, int stride, bool down, int dilation, bool avg_down = false) {
    const int out = planes * 4;
    conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(in, planes, 1).bias(false)));
    bn1 = register_module("bn1", FrozenBatchNorm2d(planes));
    conv2 = register_module("conv2", nn::Conv2d(nn::Conv2dOptions(planes, planes, 3)
                                                    .stride(stride)
                                                    .padding(dilation)
                                                    .dilation(dilation)
                                                    .bias(false)));
    bn2 = register_module("bn2", FrozenBatchNorm2d(planes));
    conv3 = register_module("conv3", nn::Conv2d(nn::Conv2dOptions(planes, out, 1).bias(false)));
    bn3 = register_module("bn3", FrozenBatchNorm2d(out));
    if (down) {
      if (avg_down && stride > 1) {
        // ResNet-D/VD: AvgPool(2,2) then a stride-1 1x1 conv (downsample.{0,1,2}).
        downsample = register_module(
            "downsample",
            nn::Sequential(nn::AvgPool2d(nn::AvgPool2dOptions(2).stride(2).ceil_mode(true)),
                           nn::Conv2d(nn::Conv2dOptions(in, out, 1).stride(1).bias(false)),
                           FrozenBatchNorm2d(out)));
      } else {
        downsample = register_module(
            "downsample",
            nn::Sequential(nn::Conv2d(nn::Conv2dOptions(in, out, 1).stride(stride).bias(false)),
                           FrozenBatchNorm2d(out)));
      }
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto identity = x;
    auto out = torch::relu(bn1->forward(conv1->forward(x)));
    out = torch::relu(bn2->forward(conv2->forward(out)));
    out = bn3->forward(conv3->forward(out));
    if (!downsample.is_empty()) {
      identity = downsample->forward(x);
    }
    return torch::relu(out + identity);
  }
};
TORCH_MODULE(Bottleneck);

// ResNet-18/34 basic block: two 3x3 convs (expansion 1) + optional downsample.
struct BasicBlockImpl : nn::Module {
  nn::Conv2d conv1{nullptr};
  nn::Conv2d conv2{nullptr};
  FrozenBatchNorm2d bn1{nullptr};
  FrozenBatchNorm2d bn2{nullptr};
  nn::Sequential downsample{nullptr};

  BasicBlockImpl(int in, int planes, int stride, bool down, int dilation) {
    conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(in, planes, 3)
                                                    .stride(stride)
                                                    .padding(dilation)
                                                    .dilation(dilation)
                                                    .bias(false)));
    bn1 = register_module("bn1", FrozenBatchNorm2d(planes));
    conv2 = register_module(
        "conv2",
        nn::Conv2d(
            nn::Conv2dOptions(planes, planes, 3).padding(dilation).dilation(dilation).bias(false)));
    bn2 = register_module("bn2", FrozenBatchNorm2d(planes));
    if (down) {
      downsample = register_module(
          "downsample",
          nn::Sequential(nn::Conv2d(nn::Conv2dOptions(in, planes, 1).stride(stride).bias(false)),
                         FrozenBatchNorm2d(planes)));
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto identity = x;
    auto out = torch::relu(bn1->forward(conv1->forward(x)));
    out = bn2->forward(conv2->forward(out));
    if (!downsample.is_empty()) {
      identity = downsample->forward(x);
    }
    return torch::relu(out + identity);
  }
};
TORCH_MODULE(BasicBlock);

nn::Sequential MakeBottleneckLayer(int in, int planes, int blocks, int stride, int dilation = 1,
                                   bool avg_down = false) {
  nn::Sequential s;
  s->push_back(Bottleneck(in, planes, stride, /*down=*/true, dilation, avg_down));
  for (int i = 1; i < blocks; ++i) {
    s->push_back(Bottleneck(planes * 4, planes, 1, false, dilation));
  }
  return s;
}

nn::Sequential MakeBasicLayer(int in, int planes, int blocks, int stride, int dilation = 1) {
  nn::Sequential s;
  const bool down = stride != 1 || in != planes;
  s->push_back(BasicBlock(in, planes, stride, down, dilation));
  for (int i = 1; i < blocks; ++i) {
    s->push_back(BasicBlock(planes, planes, 1, false, dilation));
  }
  return s;
}

}  // namespace

ResNetImpl::ResNetImpl(const std::vector<int>& blocks, bool bottleneck, bool dc5, bool deep_stem,
                       bool avg_down)
    : bottleneck_(bottleneck), deep_stem_(deep_stem) {
  if (deep_stem_) {
    // RT-DETR deep stem: 3x(3x3) conv -> 32,32,64 (first stride 2), then maxpool.
    const std::array<std::array<int, 3>, 3> cfg = {{{3, 32, 2}, {32, 32, 1}, {32, 64, 1}}};
    stem_ = register_module("stem", nn::ModuleList());
    for (const auto& [cin, cout, st] : cfg) {
      stem_->push_back(nn::Sequential(
          nn::Conv2d(nn::Conv2dOptions(cin, cout, 3).stride(st).padding(1).bias(false)),
          FrozenBatchNorm2d(cout)));
    }
  } else {
    conv1 = register_module(
        "conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)));
    bn1 = register_module("bn1", FrozenBatchNorm2d(64));
  }
  const int s4 = dc5 ? 1 : 2;
  const int d4 = dc5 ? 2 : 1;
  if (bottleneck) {
    layer1 = register_module("layer1", MakeBottleneckLayer(64, 64, blocks[0], 1, 1, avg_down));
    layer2 = register_module("layer2", MakeBottleneckLayer(256, 128, blocks[1], 2, 1, avg_down));
    layer3 = register_module("layer3", MakeBottleneckLayer(512, 256, blocks[2], 2, 1, avg_down));
    layer4 = register_module("layer4", MakeBottleneckLayer(1024, 512, blocks[3], s4, d4, avg_down));
  } else {
    layer1 = register_module("layer1", MakeBasicLayer(64, 64, blocks[0], 1));
    layer2 = register_module("layer2", MakeBasicLayer(64, 128, blocks[1], 2));
    layer3 = register_module("layer3", MakeBasicLayer(128, 256, blocks[2], 2));
    layer4 = register_module("layer4", MakeBasicLayer(256, 512, blocks[3], s4, d4));
  }
}

std::array<int, 3> ResNetImpl::feature_channels() const {
  return bottleneck_ ? std::array<int, 3>{512, 1024, 2048} : std::array<int, 3>{128, 256, 512};
}

torch::Tensor ResNetImpl::Stem(torch::Tensor x) {
  if (deep_stem_) {
    for (const auto& m : *stem_) {
      x = torch::relu(m->as<nn::SequentialImpl>()->forward(x));
    }
  } else {
    x = torch::relu(bn1->forward(conv1->forward(x)));
  }
  return torch::max_pool2d(x, {3, 3}, {2, 2}, {1, 1});
}

torch::Tensor ResNetImpl::forward(torch::Tensor x) {
  x = Stem(x);
  x = layer1->forward(x);
  x = layer2->forward(x);
  x = layer3->forward(x);
  return layer4->forward(x);
}

std::vector<torch::Tensor> ResNetImpl::forward_features(torch::Tensor x) {
  x = Stem(x);
  x = layer1->forward(x);
  auto c3 = layer2->forward(x);
  auto c4 = layer3->forward(c3);
  auto c5 = layer4->forward(c4);
  return {c3, c4, c5};
}

}  // namespace detr::models
