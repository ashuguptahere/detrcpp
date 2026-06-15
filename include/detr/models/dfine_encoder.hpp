// Copyright 2026 detrcpp authors. Apache-2.0.
//
// D-FINE HybridEncoder (neck): per-level 1x1-conv+BN input projection, an AIFI
// transformer encoder on the top level, then a CCFM top-down FPN + bottom-up PAN.
// It mirrors RT-DETR's hybrid encoder but swaps the CSP fusion block for YOLOv9's
// RepNCSPELAN4 and the PAN downsample for SCDown (spatial-channel decoupled). The
// shared primitives (ConvNorm / VGGBlock / CSPLayer / AIFI / 2D sin-cos) are
// re-implemented here with D-FINE's native key names so the authors' weights load
// 1:1. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <vector>

#include <torch/torch.h>

namespace detr::models {

namespace nn = torch::nn;

// D-FINE HybridEncoder. `in_channels`/`feat_strides` are the backbone levels feeding
// the neck (2 for nano, 3 otherwise); all outputs are `hidden_dim` wide.
struct DfHybridEncoderImpl : nn::Module {
  // `deimv2`: DEIMv2's sum-fusion FPN/PAN (RepNCSPELAN4 c1 = hidden, CSPLayer2 blocks)
  // instead of D-FINE's concat fusion. `repelan5`: the DEIMv2 RepNCSPELAN5 fuse block
  // (drops the trailing 3x3 conv per cv2/cv3 chain) used by s/m/l/x's `version=deim` neck.
  DfHybridEncoderImpl(std::vector<int> in_channels, std::vector<int> feat_strides, int hidden_dim,
                      int nhead, int dim_feedforward, double expansion, double depth_mult,
                      std::vector<int> use_encoder_idx, int num_encoder_layers, double pe_temperature,
                      bool deimv2 = false, bool repelan5 = false);
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> feats);

  std::vector<int> in_channels_, feat_strides_, use_encoder_idx_;
  int hidden_dim_;
  double pe_temperature_;
  bool deimv2_;
  nn::ModuleList input_proj{nullptr};       // per level: conv(1x1) + BN
  nn::ModuleList encoder{nullptr};          // AIFI stacks (one per use_encoder_idx)
  nn::ModuleList lateral_convs{nullptr};    // top-down 1x1 (no act)
  nn::ModuleList fpn_blocks{nullptr};       // RepNCSPELAN4
  nn::ModuleList downsample_convs{nullptr}; // SCDown
  nn::ModuleList pan_blocks{nullptr};       // RepNCSPELAN4
};
TORCH_MODULE(DfHybridEncoder);

// DEIMv2 LiteEncoder (atto/femto/pico): one backbone feature -> input_proj -> a small
// 2-scale pyramid via stride-2 avg-pool downsample + a global-average-pool fusion, then
// a sum-fusion FPN/PAN with RepNCSPELAN4 (csp2) blocks. Returns 2 levels (stride 16/32).
struct DfLiteEncoderImpl : nn::Module {
  DfLiteEncoderImpl(int in_channel, int hidden_dim, double expansion, double depth_mult);
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> feats);
  int hidden_dim_;
  nn::ModuleList input_proj{nullptr};
  nn::Sequential down_sample1{nullptr}, down_sample2{nullptr};
  nn::AnyModule bi_fusion, fpn_block, pan_block;
};
TORCH_MODULE(DfLiteEncoder);

}  // namespace detr::models
