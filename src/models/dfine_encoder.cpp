// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/dfine_encoder.hpp"

#include <cmath>
#include <cstdint>
#include <utility>

namespace detr::models {

namespace {

namespace F = torch::nn::functional;

// 2D sin-cos positional embedding for AIFI: [1, h*w, dim] (matches D-FINE's
// build_2d_sincos_position_embedding(w, h), i.e. meshgrid(arange(w), arange(h), "ij")).
torch::Tensor SinCos2d(std::int64_t h, std::int64_t w, int dim, const torch::TensorOptions& opts,
                       double temp) {
  auto gw = torch::arange(w, opts);
  auto gh = torch::arange(h, opts);
  auto grid = torch::meshgrid({gw, gh}, "ij");
  const int pos_dim = dim / 4;
  auto omega = torch::arange(pos_dim, opts) / static_cast<double>(pos_dim);
  omega = 1.0 / torch::pow(temp, omega);
  auto out_w = grid[0].flatten().unsqueeze(1) * omega.unsqueeze(0);
  auto out_h = grid[1].flatten().unsqueeze(1) * omega.unsqueeze(0);
  auto pos = torch::cat({out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()}, 1);
  return pos.unsqueeze(0);
}

// Conv2d(no bias) + BatchNorm + optional SiLU. (D-FINE ConvNormLayer / *_fuse, non-deploy.)
struct DfConvNormImpl : nn::Module {
  nn::Conv2d conv{nullptr};
  nn::BatchNorm2d norm{nullptr};
  bool act_;
  DfConvNormImpl(int in, int out, int k, int s, bool act, int groups = 1) : act_(act) {
    conv = register_module("conv", nn::Conv2d(nn::Conv2dOptions(in, out, k)
                                                  .stride(s)
                                                  .padding((k - 1) / 2)
                                                  .groups(groups)
                                                  .bias(false)));
    norm = register_module("norm", nn::BatchNorm2d(out));
  }
  torch::Tensor forward(torch::Tensor x) {
    x = norm->forward(conv->forward(x));
    return act_ ? torch::silu(x) : x;
  }
};
TORCH_MODULE(DfConvNorm);

// VGGBlock: 3x3 + 1x1 (each no act), summed, SiLU.
struct DfVggImpl : nn::Module {
  DfConvNorm conv1{nullptr}, conv2{nullptr};
  DfVggImpl(int c) {
    conv1 = register_module("conv1", DfConvNorm(c, c, 3, 1, false));
    conv2 = register_module("conv2", DfConvNorm(c, c, 1, 1, false));
  }
  torch::Tensor forward(torch::Tensor x) {
    return torch::silu(conv1->forward(x) + conv2->forward(x));
  }
};
TORCH_MODULE(DfVgg);

// CSPLayer: two 1x1 branches to a hidden width, VGG bottlenecks on one, summed, then a
// fuse 1x1 back to out (Identity when hidden == out).
struct DfCSPLayerImpl : nn::Module {
  DfConvNorm conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
  nn::Sequential bottlenecks{nullptr};
  DfCSPLayerImpl(int in, int out, int num_blocks, double expansion = 1.0) {
    const int hidden = static_cast<int>(static_cast<double>(out) * expansion);
    conv1 = register_module("conv1", DfConvNorm(in, hidden, 1, 1, true));
    conv2 = register_module("conv2", DfConvNorm(in, hidden, 1, 1, true));
    bottlenecks = nn::Sequential();
    for (int i = 0; i < num_blocks; ++i) {
      bottlenecks->push_back(DfVgg(hidden));
    }
    register_module("bottlenecks", bottlenecks);
    if (hidden != out) {
      conv3 = register_module("conv3", DfConvNorm(hidden, out, 1, 1, true));
    }
  }
  torch::Tensor forward(torch::Tensor x) {
    auto y = bottlenecks->forward(conv1->forward(x)) + conv2->forward(x);
    return conv3.is_empty() ? y : conv3->forward(y);
  }
};
TORCH_MODULE(DfCSPLayer);

// CSPLayer2 (DEIMv2 "csp2", RepC3-style): conv1 -> 2*hidden, split, y0 + bottlenecks(y1),
// fused by conv3 (Identity when hidden == out). No second 1x1 branch.
struct DfCSPLayer2Impl : nn::Module {
  DfConvNorm conv1{nullptr}, conv3{nullptr};
  nn::Sequential bottlenecks{nullptr};
  DfCSPLayer2Impl(int in, int out, int num_blocks, double expansion = 1.0) {
    const int hidden = static_cast<int>(static_cast<double>(out) * expansion);
    conv1 = register_module("conv1", DfConvNorm(in, 2 * hidden, 1, 1, true));
    bottlenecks = nn::Sequential();
    for (int i = 0; i < num_blocks; ++i) {
      bottlenecks->push_back(DfVgg(hidden));
    }
    register_module("bottlenecks", bottlenecks);
    if (hidden != out) {
      conv3 = register_module("conv3", DfConvNorm(hidden, out, 1, 1, true));
    }
  }
  torch::Tensor forward(torch::Tensor x) {
    auto y = conv1->forward(x).chunk(2, 1);
    auto fused = y[0] + bottlenecks->forward(y[1]);
    return conv3.is_empty() ? fused : conv3->forward(fused);
  }
};
TORCH_MODULE(DfCSPLayer2);

// RepNCSPELAN4 (YOLOv9 CSP-ELAN): split cv1 output, run two CSPLayer->3x3 chains off the
// second half, concat all four, fuse with cv4. `csp2` selects DEIMv2's CSPLayer2.
struct DfRepNCSPELAN4Impl : nn::Module {
  DfConvNorm cv1{nullptr}, cv4{nullptr};
  nn::Sequential cv2{nullptr}, cv3{nullptr};
  int c_;
  DfRepNCSPELAN4Impl(int c1, int c2, int c3, int c4, int n, bool csp2 = false) : c_(c3 / 2) {
    cv1 = register_module("cv1", DfConvNorm(c1, c3, 1, 1, true));
    if (csp2) {
      cv2 = register_module("cv2", nn::Sequential(DfCSPLayer2(c3 / 2, c4, n), DfConvNorm(c4, c4, 3, 1, true)));
      cv3 = register_module("cv3", nn::Sequential(DfCSPLayer2(c4, c4, n), DfConvNorm(c4, c4, 3, 1, true)));
    } else {
      cv2 = register_module("cv2", nn::Sequential(DfCSPLayer(c3 / 2, c4, n), DfConvNorm(c4, c4, 3, 1, true)));
      cv3 = register_module("cv3", nn::Sequential(DfCSPLayer(c4, c4, n), DfConvNorm(c4, c4, 3, 1, true)));
    }
    cv4 = register_module("cv4", DfConvNorm(c3 + 2 * c4, c2, 1, 1, true));
  }
  torch::Tensor forward(torch::Tensor x) {
    auto y = cv1->forward(x).split(c_, 1);  // [y0, y1] each c_
    auto y2 = cv2->forward(y[1]);
    auto y3 = cv3->forward(y2);
    return cv4->forward(torch::cat({y[0], y[1], y2, y3}, 1));
  }
};
TORCH_MODULE(DfRepNCSPELAN4);

// SCDown: 1x1 channel mix then depthwise kxk stride-s downsample (both no act).
struct DfSCDownImpl : nn::Module {
  DfConvNorm cv1{nullptr}, cv2{nullptr};
  DfSCDownImpl(int c1, int c2, int k, int s) {
    cv1 = register_module("cv1", DfConvNorm(c1, c2, 1, 1, false));
    cv2 = register_module("cv2", DfConvNorm(c2, c2, k, s, false, /*groups=*/c2));
  }
  torch::Tensor forward(torch::Tensor x) { return cv2->forward(cv1->forward(x)); }
};
TORCH_MODULE(DfSCDown);

// AIFI: post-norm transformer encoder layer, GELU FFN (D-FINE enc_act="gelu").
struct DfAIFIImpl : nn::Module {
  nn::MultiheadAttention self_attn{nullptr};
  nn::Linear linear1{nullptr}, linear2{nullptr};
  nn::LayerNorm norm1{nullptr}, norm2{nullptr};
  DfAIFIImpl(int d, int heads, int ff) {
    self_attn = register_module(
        "self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, heads).dropout(0.0)));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
  }
  torch::Tensor forward(torch::Tensor src, const torch::Tensor& pos) {
    auto q = (src + pos).transpose(0, 1);  // [L, N, d]
    auto k = q;
    auto v = src.transpose(0, 1);
    auto attn = std::get<0>(self_attn->forward(q, k, v)).transpose(0, 1);
    src = norm1->forward(src + attn);
    auto ff = linear2->forward(torch::gelu(linear1->forward(src)));
    return norm2->forward(src + ff);
  }
};
TORCH_MODULE(DfAIFI);

// One AIFI stack ("layers" of TransformerEncoderLayer); a TransformerEncoder.
struct DfEncStackImpl : nn::Module {
  nn::ModuleList layers{nullptr};
  DfEncStackImpl(int d, int heads, int ff, int num_layers) {
    layers = register_module("layers", nn::ModuleList());
    for (int i = 0; i < num_layers; ++i) {
      layers->push_back(DfAIFI(d, heads, ff));
    }
  }
  torch::Tensor forward(torch::Tensor src, const torch::Tensor& pos) {
    for (const auto& l : *layers) {
      src = l->as<DfAIFIImpl>()->forward(src, pos);
    }
    return src;
  }
};
TORCH_MODULE(DfEncStack);

// Per-level input projection: 1x1 conv (no bias) + BN, no activation.
struct DfInputProjImpl : nn::Module {
  nn::Conv2d conv{nullptr};
  nn::BatchNorm2d norm{nullptr};
  DfInputProjImpl(int in, int out) {
    conv = register_module("conv", nn::Conv2d(nn::Conv2dOptions(in, out, 1).bias(false)));
    norm = register_module("norm", nn::BatchNorm2d(out));
  }
  torch::Tensor forward(torch::Tensor x) { return norm->forward(conv->forward(x)); }
};
TORCH_MODULE(DfInputProj);

// GAP_Fusion (DEIMv2 LiteEncoder): add the global-average-pooled context, then a 1x1 ConvNorm.
struct DfGapFusionImpl : nn::Module {
  DfConvNorm cv{nullptr};
  explicit DfGapFusionImpl(int ch) { cv = register_module("cv", DfConvNorm(ch, ch, 1, 1, true)); }
  torch::Tensor forward(torch::Tensor x) {
    return cv->forward(x + torch::adaptive_avg_pool2d(x, {1, 1}));
  }
};
TORCH_MODULE(DfGapFusion);

// Stride-2 avg-pool downsample: AvgPool(3,2,1) -> 1x1 conv(no bias) -> BN -> SiLU.
nn::Sequential MakeLiteDown(int ch) {
  return nn::Sequential(nn::AvgPool2d(nn::AvgPool2dOptions(3).stride(2).padding(1)),
                        nn::Conv2d(nn::Conv2dOptions(ch, ch, 1).bias(false)), nn::BatchNorm2d(ch),
                        nn::Functional(torch::silu));
}

}  // namespace

DfHybridEncoderImpl::DfHybridEncoderImpl(std::vector<int> in_channels, std::vector<int> feat_strides,
                                         int hidden_dim, int nhead, int dim_feedforward,
                                         double expansion, double depth_mult,
                                         std::vector<int> use_encoder_idx, int num_encoder_layers,
                                         double pe_temperature, bool deimv2)
    : in_channels_(std::move(in_channels)),
      feat_strides_(std::move(feat_strides)),
      use_encoder_idx_(std::move(use_encoder_idx)),
      hidden_dim_(hidden_dim),
      pe_temperature_(pe_temperature),
      deimv2_(deimv2) {
  const int L = static_cast<int>(in_channels_.size());
  // DEIMv2 sums upsample+feat_low (block c1 = hidden_dim) and uses CSPLayer2 blocks.
  const int fuse_in = deimv2 ? hidden_dim : hidden_dim * 2;
  // RepNCSPELAN4 hidden width and depth, matching D-FINE's round((expansion*hidden)//2)
  // and round(3*depth_mult).
  const int c4 = static_cast<int>(std::floor(expansion * hidden_dim / 2.0));
  const int nblk = static_cast<int>(std::lround(3.0 * depth_mult));

  input_proj = register_module("input_proj", nn::ModuleList());
  for (int ch : in_channels_) {
    input_proj->push_back(DfInputProj(ch, hidden_dim));
  }
  encoder = register_module("encoder", nn::ModuleList());
  for (std::size_t i = 0; i < use_encoder_idx_.size(); ++i) {
    encoder->push_back(DfEncStack(hidden_dim, nhead, dim_feedforward, num_encoder_layers));
  }
  lateral_convs = register_module("lateral_convs", nn::ModuleList());
  fpn_blocks = register_module("fpn_blocks", nn::ModuleList());
  for (int i = 0; i < L - 1; ++i) {
    lateral_convs->push_back(DfConvNorm(hidden_dim, hidden_dim, 1, 1, false));
    fpn_blocks->push_back(DfRepNCSPELAN4(fuse_in, hidden_dim, hidden_dim * 2, c4, nblk, deimv2));
  }
  downsample_convs = register_module("downsample_convs", nn::ModuleList());
  pan_blocks = register_module("pan_blocks", nn::ModuleList());
  for (int i = 0; i < L - 1; ++i) {
    downsample_convs->push_back(DfSCDown(hidden_dim, hidden_dim, 3, 2));
    pan_blocks->push_back(DfRepNCSPELAN4(fuse_in, hidden_dim, hidden_dim * 2, c4, nblk, deimv2));
  }
}

std::vector<torch::Tensor> DfHybridEncoderImpl::forward(std::vector<torch::Tensor> feats) {
  const int L = static_cast<int>(in_channels_.size());
  std::vector<torch::Tensor> proj;
  proj.reserve(feats.size());
  for (std::size_t i = 0; i < feats.size(); ++i) {
    proj.push_back(input_proj[i]->as<DfInputProjImpl>()->forward(feats[i]));
  }
  // AIFI on the selected level(s).
  for (std::size_t i = 0; i < use_encoder_idx_.size(); ++i) {
    const int e = use_encoder_idx_[i];
    const auto h = proj[static_cast<std::size_t>(e)].size(2);
    const auto w = proj[static_cast<std::size_t>(e)].size(3);
    auto src = proj[static_cast<std::size_t>(e)].flatten(2).permute({0, 2, 1});  // [B,hw,C]
    auto pos = SinCos2d(h, w, hidden_dim_, src.options(), pe_temperature_);
    auto mem = encoder[i]->as<DfEncStackImpl>()->forward(src, pos);
    proj[static_cast<std::size_t>(e)] =
        mem.permute({0, 2, 1}).reshape({-1, hidden_dim_, h, w}).contiguous();
  }
  // Top-down FPN.
  std::vector<torch::Tensor> inner{proj[static_cast<std::size_t>(L - 1)]};
  for (int idx = L - 1; idx > 0; --idx) {
    auto feat_high = inner.front();
    auto feat_low = proj[static_cast<std::size_t>(idx - 1)];
    feat_high = lateral_convs[L - 1 - idx]->as<DfConvNormImpl>()->forward(feat_high);
    inner.front() = feat_high;
    auto up = F::interpolate(feat_high,
                             F::InterpolateFuncOptions().scale_factor(std::vector<double>{2.0, 2.0}).mode(
                                 torch::kNearest));
    auto fused = deimv2_ ? (up + feat_low) : torch::cat({up, feat_low}, 1);
    auto inner_out = fpn_blocks[L - 1 - idx]->as<DfRepNCSPELAN4Impl>()->forward(fused);
    inner.insert(inner.begin(), inner_out);
  }
  // Bottom-up PAN.
  std::vector<torch::Tensor> outs{inner.front()};
  for (int idx = 0; idx < L - 1; ++idx) {
    auto feat_low = outs.back();
    auto feat_high = inner[static_cast<std::size_t>(idx + 1)];
    auto down = downsample_convs[idx]->as<DfSCDownImpl>()->forward(feat_low);
    auto fused = deimv2_ ? (down + feat_high) : torch::cat({down, feat_high}, 1);
    outs.push_back(pan_blocks[idx]->as<DfRepNCSPELAN4Impl>()->forward(fused));
  }
  return outs;
}

DfLiteEncoderImpl::DfLiteEncoderImpl(int in_channel, int hidden_dim, double expansion,
                                     double depth_mult)
    : hidden_dim_(hidden_dim) {
  const int c4 = static_cast<int>(std::floor(expansion * hidden_dim / 2.0));
  const int nblk = static_cast<int>(std::lround(3.0 * depth_mult));
  input_proj = register_module("input_proj", nn::ModuleList());
  input_proj->push_back(DfInputProj(in_channel, hidden_dim));
  down_sample1 = register_module("down_sample1", MakeLiteDown(hidden_dim));
  down_sample2 = register_module("down_sample2", MakeLiteDown(hidden_dim));
  auto bf = DfGapFusion(hidden_dim);
  register_module("bi_fusion", bf);
  bi_fusion = nn::AnyModule(bf);
  auto fpn = DfRepNCSPELAN4(hidden_dim, hidden_dim, hidden_dim * 2, c4, nblk, /*csp2=*/true);
  register_module("fpn_block", fpn);
  fpn_block = nn::AnyModule(fpn);
  auto pan = DfRepNCSPELAN4(hidden_dim, hidden_dim, hidden_dim * 2, c4, nblk, /*csp2=*/true);
  register_module("pan_block", pan);
  pan_block = nn::AnyModule(pan);
}

std::vector<torch::Tensor> DfLiteEncoderImpl::forward(std::vector<torch::Tensor> feats) {
  auto proj = input_proj[0]->as<DfInputProjImpl>()->forward(feats[0]);  // [B, hidden, H/16, W/16]
  auto ds1 = down_sample1->forward(proj);                               // [B, hidden, H/32, W/32]
  ds1 = bi_fusion.forward<torch::Tensor>(ds1);
  auto up = F::interpolate(
      ds1, F::InterpolateFuncOptions().scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest));
  auto out0 = fpn_block.forward<torch::Tensor>(proj + up);
  auto out1 = pan_block.forward<torch::Tensor>(ds1 + down_sample2->forward(out0));
  return {out0, out1};
}

}  // namespace detr::models
