// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/hgnetv2.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <utility>

namespace detr::models {

namespace F = torch::nn::functional;

HgLabImpl::HgLabImpl() {
  scale = register_parameter("scale", torch::ones({1}));
  bias = register_parameter("bias", torch::zeros({1}));
}

torch::Tensor HgLabImpl::forward(const torch::Tensor& x) { return scale * x + bias; }

HgConvImpl::HgConvImpl(int in_ch, int out_ch, int kernel, int stride, int groups, bool use_act,
                       bool use_lab)
    : use_act_(use_act) {
  conv = register_module("conv", nn::Conv2d(nn::Conv2dOptions(in_ch, out_ch, kernel)
                                                .stride(stride)
                                                .padding((kernel - 1) / 2)
                                                .groups(groups)
                                                .bias(false)));
  bn = register_module("bn", FrozenBatchNorm2d(out_ch));
  if (use_act_ && use_lab) {
    lab = register_module("lab", HgLab());
  }
}

torch::Tensor HgConvImpl::forward(torch::Tensor x) {
  x = bn->forward(conv->forward(x));
  if (use_act_) {
    x = torch::relu(x);
  }
  if (lab) {
    x = lab->forward(x);
  }
  return x;
}

HgLightConvImpl::HgLightConvImpl(int in_ch, int out_ch, int kernel, bool use_lab) {
  conv1 = register_module("conv1", HgConv(in_ch, out_ch, 1, 1, 1, /*use_act=*/false, use_lab));
  conv2 = register_module("conv2",
                          HgConv(out_ch, out_ch, kernel, 1, /*groups=*/out_ch, /*use_act=*/true, use_lab));
}

torch::Tensor HgLightConvImpl::forward(torch::Tensor x) {
  return conv2->forward(conv1->forward(x));
}

HgStemImpl::HgStemImpl(int in_ch, int mid_ch, int out_ch, bool use_lab) {
  stem1 = register_module("stem1", HgConv(in_ch, mid_ch, 3, 2, 1, true, use_lab));
  stem2a = register_module("stem2a", HgConv(mid_ch, mid_ch / 2, 2, 1, 1, true, use_lab));
  stem2b = register_module("stem2b", HgConv(mid_ch / 2, mid_ch, 2, 1, 1, true, use_lab));
  stem3 = register_module("stem3", HgConv(mid_ch * 2, mid_ch, 3, 2, 1, true, use_lab));
  stem4 = register_module("stem4", HgConv(mid_ch, out_ch, 1, 1, 1, true, use_lab));
  pool = register_module("pool", nn::MaxPool2d(nn::MaxPool2dOptions(2).stride(1).ceil_mode(true)));
}

torch::Tensor HgStemImpl::forward(torch::Tensor x) {
  x = stem1->forward(x);
  x = F::pad(x, F::PadFuncOptions({0, 1, 0, 1}));  // (left,right,top,bottom)
  auto x2 = stem2a->forward(x);
  x2 = F::pad(x2, F::PadFuncOptions({0, 1, 0, 1}));
  x2 = stem2b->forward(x2);
  auto x1 = pool->forward(x);
  x = torch::cat({x1, x2}, 1);
  x = stem3->forward(x);
  x = stem4->forward(x);
  return x;
}

HgBlockImpl::HgBlockImpl(int in_ch, int mid_ch, int out_ch, int layer_num, int kernel, bool residual,
                         bool light, bool use_lab)
    : residual_(residual), light_(light) {
  layers = register_module("layers", nn::ModuleList());
  for (int i = 0; i < layer_num; ++i) {
    const int lin = (i == 0) ? in_ch : mid_ch;
    if (light_) {
      layers->push_back(HgLightConv(lin, mid_ch, kernel, use_lab));
    } else {
      layers->push_back(HgConv(lin, mid_ch, kernel, 1, 1, true, use_lab));
    }
  }
  // "se" aggregation: squeeze (total -> out/2) then excite (out/2 -> out).
  const int total = in_ch + layer_num * mid_ch;
  aggregation = register_module("aggregation", nn::ModuleList());
  aggregation->push_back(HgConv(total, out_ch / 2, 1, 1, 1, true, use_lab));
  aggregation->push_back(HgConv(out_ch / 2, out_ch, 1, 1, 1, true, use_lab));
}

torch::Tensor HgBlockImpl::forward(torch::Tensor x) {
  auto identity = x;
  std::vector<torch::Tensor> outs{x};
  for (const auto& m : *layers) {
    x = light_ ? m->as<HgLightConvImpl>()->forward(x) : m->as<HgConvImpl>()->forward(x);
    outs.push_back(x);
  }
  x = torch::cat(outs, 1);
  for (const auto& m : *aggregation) {
    x = m->as<HgConvImpl>()->forward(x);
  }
  if (residual_) {
    x = x + identity;
  }
  return x;
}

HgStageImpl::HgStageImpl(int in_ch, int mid_ch, int out_ch, int block_num, int layer_num,
                         bool has_downsample, bool light, int kernel, bool use_lab) {
  if (has_downsample) {
    // Depthwise 3x3 stride-2, no activation (so no LAB either).
    downsample = register_module(
        "downsample", HgConv(in_ch, in_ch, 3, 2, /*groups=*/in_ch, /*use_act=*/false, use_lab));
  }
  blocks = register_module("blocks", nn::ModuleList());
  for (int i = 0; i < block_num; ++i) {
    const int bin = (i == 0) ? in_ch : out_ch;
    blocks->push_back(
        HgBlock(bin, mid_ch, out_ch, layer_num, kernel, /*residual=*/i != 0, light, use_lab));
  }
}

torch::Tensor HgStageImpl::forward(torch::Tensor x) {
  if (downsample) {
    x = downsample->forward(x);
  }
  for (const auto& b : *blocks) {
    x = b->as<HgBlockImpl>()->forward(x);
  }
  return x;
}

namespace {

// {in, mid, out, num_blocks, downsample, light_block, kernel, layer_num} per stage.
struct StageCfg {
  int in_ch, mid_ch, out_ch, blocks, layer_num, kernel;
  bool downsample, light;
};
struct ArchCfg {
  std::array<int, 3> stem;  // {in, mid, out}
  std::vector<StageCfg> stages;
};

// HGNetv2 B0/B2/B4/B5 (D-FINE n,s / m / l / x). From PPHGNetV2 arch_configs.
ArchCfg Arch(const std::string& v) {
  static const std::map<std::string, ArchCfg> kArch = {
      {"B0",
       {{3, 16, 16},
        {{16, 16, 64, 1, 3, 3, false, false},
         {64, 32, 256, 1, 3, 3, true, false},
         {256, 64, 512, 2, 3, 5, true, true},
         {512, 128, 1024, 1, 3, 5, true, true}}}},
      {"B2",
       {{3, 24, 32},
        {{32, 32, 96, 1, 4, 3, false, false},
         {96, 64, 384, 1, 4, 3, true, false},
         {384, 128, 768, 3, 4, 5, true, true},
         {768, 256, 1536, 1, 4, 5, true, true}}}},
      {"B4",
       {{3, 32, 48},
        {{48, 48, 128, 1, 6, 3, false, false},
         {128, 96, 512, 1, 6, 3, true, false},
         {512, 192, 1024, 3, 6, 5, true, true},
         {1024, 384, 2048, 1, 6, 5, true, true}}}},
      {"B5",
       {{3, 32, 64},
        {{64, 64, 128, 1, 6, 3, false, false},
         {128, 128, 512, 2, 6, 3, true, false},
         {512, 256, 1024, 5, 6, 5, true, true},
         {1024, 512, 2048, 2, 6, 5, true, true}}}},
      // DEIMv2 micro variants: 3 stages only (Atto/Femto/Pico). {in,mid,out,blocks,layer,kernel,down,light}.
      {"Atto",
       {{3, 16, 16},
        {{16, 16, 64, 1, 3, 3, false, false},
         {64, 32, 256, 1, 3, 3, true, false},
         {256, 64, 256, 1, 3, 3, true, true}}}},
      {"Femto",
       {{3, 16, 16},
        {{16, 16, 64, 1, 3, 3, false, false},
         {64, 32, 256, 1, 3, 3, true, false},
         {256, 64, 512, 1, 3, 5, true, true}}}},
      {"Pico",
       {{3, 16, 16},
        {{16, 16, 64, 1, 3, 3, false, false},
         {64, 32, 256, 1, 3, 3, true, false},
         {256, 64, 512, 2, 3, 5, true, true}}}},
  };
  return kArch.at(v);
}

}  // namespace

HgNetV2Impl::HgNetV2Impl(const std::string& variant, bool use_lab, std::vector<int> return_idx)
    : return_idx_(std::move(return_idx)) {
  const ArchCfg cfg = Arch(variant);
  stem = register_module("stem", HgStem(cfg.stem[0], cfg.stem[1], cfg.stem[2], use_lab));
  stages = register_module("stages", nn::ModuleList());
  for (std::size_t i = 0; i < cfg.stages.size(); ++i) {
    const StageCfg& s = cfg.stages[i];
    stages->push_back(HgStage(s.in_ch, s.mid_ch, s.out_ch, s.blocks, s.layer_num, s.downsample,
                              s.light, s.kernel, use_lab));
    for (int r : return_idx_) {
      if (r == static_cast<int>(i)) {
        out_channels_.push_back(s.out_ch);
      }
    }
  }
}

std::vector<torch::Tensor> HgNetV2Impl::forward(torch::Tensor x) {
  x = stem->forward(x);
  std::vector<torch::Tensor> outs;
  for (std::size_t i = 0; i < stages->size(); ++i) {
    x = stages[i]->as<HgStageImpl>()->forward(x);
    for (int r : return_idx_) {
      if (r == static_cast<int>(i)) {
        outs.push_back(x);
      }
    }
  }
  return outs;
}

}  // namespace detr::models
