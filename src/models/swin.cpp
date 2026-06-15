// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/swin.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace detr::models {

namespace {

using torch::indexing::Slice;

// Partition [B, H, W, C] into non-overlapping windows -> [nW*B, ws, ws, C].
torch::Tensor WindowPartition(const torch::Tensor& x, int ws) {
  const auto B = x.size(0), H = x.size(1), W = x.size(2), C = x.size(3);
  auto v = x.view({B, H / ws, ws, W / ws, ws, C});
  return v.permute({0, 1, 3, 2, 4, 5}).contiguous().view({-1, ws, ws, C});
}

// Inverse of WindowPartition: [nW*B, ws, ws, C] -> [B, H, W, C].
torch::Tensor WindowReverse(const torch::Tensor& windows, int ws, int H, int W) {
  const auto B = windows.size(0) / (static_cast<std::int64_t>(H) * W / ws / ws);
  auto x = windows.view({B, H / ws, W / ws, ws, ws, -1});
  return x.permute({0, 1, 3, 2, 4, 5}).contiguous().view({B, H, W, -1});
}

// Pair-wise relative-position index for a ws x ws window: [ws*ws, ws*ws] (long).
torch::Tensor RelativePositionIndex(int ws) {
  auto coords_h = torch::arange(ws, torch::kLong);
  auto coords_w = torch::arange(ws, torch::kLong);
  auto hh = coords_h.view({ws, 1}).expand({ws, ws});
  auto ww = coords_w.view({1, ws}).expand({ws, ws});
  auto coords = torch::stack({hh, ww}, 0).flatten(1);             // [2, ws*ws]
  auto rel = coords.unsqueeze(2) - coords.unsqueeze(1);           // [2, N, N]
  rel = rel.permute({1, 2, 0}).contiguous();                      // [N, N, 2]
  rel.select(2, 0) += ws - 1;
  rel.select(2, 1) += ws - 1;
  rel.select(2, 0) *= 2 * ws - 1;
  return rel.sum(-1);                                             // [N, N]
}

}  // namespace

SwinPatchEmbedImpl::SwinPatchEmbedImpl(int patch_size, int in_chans, int embed_dim, bool with_norm)
    : patch_size_(patch_size), embed_dim_(embed_dim) {
  proj = register_module(
      "proj", nn::Conv2d(nn::Conv2dOptions(in_chans, embed_dim, patch_size).stride(patch_size)));
  if (with_norm) norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({embed_dim})));
}

torch::Tensor SwinPatchEmbedImpl::forward(torch::Tensor x) {
  // Pad H/W up to a patch-size multiple (right / bottom only, matching the reference).
  const auto H = x.size(2), W = x.size(3);
  if (W % patch_size_ != 0) {
    x = torch::constant_pad_nd(x, {0, patch_size_ - W % patch_size_}, 0);
  }
  if (H % patch_size_ != 0) {
    x = torch::constant_pad_nd(x, {0, 0, 0, patch_size_ - H % patch_size_}, 0);
  }
  x = proj->forward(x);  // [B, C, Wh, Ww]
  if (norm) {
    const auto Wh = x.size(2), Ww = x.size(3);
    x = x.flatten(2).transpose(1, 2);  // [B, Wh*Ww, C]
    x = norm->forward(x);
    x = x.transpose(1, 2).view({-1, embed_dim_, Wh, Ww});
  }
  return x;
}

SwinWindowAttentionImpl::SwinWindowAttentionImpl(int dim, int window_size, int num_heads)
    : window_size_(window_size),
      num_heads_(num_heads),
      scale_(std::pow(static_cast<double>(dim / num_heads), -0.5)) {
  const int n = (2 * window_size - 1) * (2 * window_size - 1);
  relative_position_bias_table =
      register_parameter("relative_position_bias_table", torch::zeros({n, num_heads}));
  relative_position_index =
      register_buffer("relative_position_index", RelativePositionIndex(window_size));
  qkv = register_module("qkv", nn::Linear(dim, dim * 3));
  proj = register_module("proj", nn::Linear(dim, dim));
}

torch::Tensor SwinWindowAttentionImpl::forward(torch::Tensor x, const torch::Tensor& mask) {
  const auto B = x.size(0), N = x.size(1), C = x.size(2);
  auto qkv_out =
      qkv->forward(x).reshape({B, N, 3, num_heads_, C / num_heads_}).permute({2, 0, 3, 1, 4});
  auto q = qkv_out[0] * scale_, k = qkv_out[1], v = qkv_out[2];  // each [B, nH, N, head_dim]
  auto attn = torch::matmul(q, k.transpose(-2, -1));             // [B, nH, N, N]

  const auto M = window_size_ * window_size_;
  auto bias = relative_position_bias_table.index_select(0, relative_position_index.view(-1))
                  .view({M, M, -1})
                  .permute({2, 0, 1})
                  .contiguous();  // [nH, N, N]
  attn = attn + bias.unsqueeze(0);

  if (mask.defined()) {
    const auto nW = mask.size(0);
    attn = attn.view({B / nW, nW, num_heads_, N, N}) + mask.unsqueeze(1).unsqueeze(0);
    attn = attn.view({-1, num_heads_, N, N});
  }
  attn = torch::softmax(attn, -1);
  auto o = torch::matmul(attn, v).transpose(1, 2).reshape({B, N, C});
  return proj->forward(o);
}

SwinBlockImpl::SwinBlockImpl(int dim, int num_heads, int window_size, int shift_size,
                             double mlp_ratio)
    : window_size_(window_size), shift_size_(shift_size) {
  norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({dim})));
  attn = register_module("attn", SwinWindowAttention(dim, window_size, num_heads));
  norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({dim})));
  auto mlp = register_module("mlp", std::make_shared<nn::Module>());
  const int hidden = static_cast<int>(dim * mlp_ratio);
  fc1 = mlp->register_module("fc1", nn::Linear(dim, hidden));
  fc2 = mlp->register_module("fc2", nn::Linear(hidden, dim));
}

torch::Tensor SwinBlockImpl::forward(torch::Tensor x, int H, int W, const torch::Tensor& mask) {
  const auto B = x.size(0), C = x.size(2);
  auto shortcut = x;
  x = norm1->forward(x).view({B, H, W, C});

  // Pad to a window multiple (right / bottom).
  const int pad_r = (window_size_ - W % window_size_) % window_size_;
  const int pad_b = (window_size_ - H % window_size_) % window_size_;
  x = torch::constant_pad_nd(x, {0, 0, 0, pad_r, 0, pad_b}, 0);
  const int Hp = static_cast<int>(x.size(1)), Wp = static_cast<int>(x.size(2));

  torch::Tensor attn_mask;
  if (shift_size_ > 0) {
    x = torch::roll(x, {-shift_size_, -shift_size_}, {1, 2});
    attn_mask = mask;
  }

  auto x_windows = WindowPartition(x, window_size_).view({-1, window_size_ * window_size_, C});
  auto attn_windows = attn->forward(x_windows, attn_mask);
  attn_windows = attn_windows.view({-1, window_size_, window_size_, C});
  x = WindowReverse(attn_windows, window_size_, Hp, Wp);  // [B, Hp, Wp, C]

  if (shift_size_ > 0) x = torch::roll(x, {shift_size_, shift_size_}, {1, 2});
  if (pad_r > 0 || pad_b > 0) x = x.index({Slice(), Slice(0, H), Slice(0, W), Slice()}).contiguous();
  x = x.view({B, static_cast<std::int64_t>(H) * W, C});

  x = shortcut + x;
  x = x + fc2->forward(torch::gelu(fc1->forward(norm2->forward(x))));
  return x;
}

SwinPatchMergingImpl::SwinPatchMergingImpl(int dim) {
  reduction = register_module("reduction", nn::Linear(nn::LinearOptions(4 * dim, 2 * dim).bias(false)));
  norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({4 * dim})));
}

torch::Tensor SwinPatchMergingImpl::forward(torch::Tensor x, int H, int W) {
  const auto B = x.size(0), C = x.size(2);
  x = x.view({B, H, W, C});
  // Pad to even H/W before the 2x2 strided gather.
  if (H % 2 == 1 || W % 2 == 1) x = torch::constant_pad_nd(x, {0, 0, 0, W % 2, 0, H % 2}, 0);

  auto x0 = x.index({Slice(), Slice(0, torch::indexing::None, 2), Slice(0, torch::indexing::None, 2), Slice()});
  auto x1 = x.index({Slice(), Slice(1, torch::indexing::None, 2), Slice(0, torch::indexing::None, 2), Slice()});
  auto x2 = x.index({Slice(), Slice(0, torch::indexing::None, 2), Slice(1, torch::indexing::None, 2), Slice()});
  auto x3 = x.index({Slice(), Slice(1, torch::indexing::None, 2), Slice(1, torch::indexing::None, 2), Slice()});
  x = torch::cat({x0, x1, x2, x3}, -1).view({B, -1, 4 * C});  // [B, H/2*W/2, 4C]
  return reduction->forward(norm->forward(x));
}

SwinBasicLayerImpl::SwinBasicLayerImpl(int dim, int depth, int num_heads, int window_size,
                                       double mlp_ratio, bool with_downsample)
    : window_size_(window_size), shift_size_(window_size / 2) {
  blocks = register_module("blocks", nn::ModuleList());
  for (int i = 0; i < depth; ++i) {
    blocks->push_back(
        SwinBlock(dim, num_heads, window_size, (i % 2 == 0) ? 0 : window_size / 2, mlp_ratio));
  }
  if (with_downsample) downsample = register_module("downsample", SwinPatchMerging(dim));
}

SwinStageOut SwinBasicLayerImpl::forward(torch::Tensor x, int H, int W) {
  // SW-MSA attention mask: window indices over a cyclically-shifted image grid.
  const int Hp = static_cast<int>(std::ceil(static_cast<double>(H) / window_size_)) * window_size_;
  const int Wp = static_cast<int>(std::ceil(static_cast<double>(W) / window_size_)) * window_size_;
  auto img_mask = torch::zeros({1, Hp, Wp, 1}, x.options());
  const std::vector<std::pair<int, int>> h_seg = {
      {0, Hp - window_size_}, {Hp - window_size_, Hp - shift_size_}, {Hp - shift_size_, Hp}};
  const std::vector<std::pair<int, int>> w_seg = {
      {0, Wp - window_size_}, {Wp - window_size_, Wp - shift_size_}, {Wp - shift_size_, Wp}};
  int cnt = 0;
  for (const auto& hs : h_seg) {
    for (const auto& wsg : w_seg) {
      img_mask.index_put_({Slice(), Slice(hs.first, hs.second), Slice(wsg.first, wsg.second), Slice()},
                          static_cast<double>(cnt));
      ++cnt;
    }
  }
  auto mask_windows = WindowPartition(img_mask, window_size_).view({-1, window_size_ * window_size_});
  auto attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2);
  auto nz = attn_mask.ne(0);
  attn_mask = attn_mask.masked_fill(nz, -100.0).masked_fill(nz.logical_not(), 0.0);

  for (std::size_t i = 0; i < blocks->size(); ++i) {
    x = blocks[i]->as<SwinBlockImpl>()->forward(x, H, W, attn_mask);
  }
  if (downsample) {
    auto down = downsample->forward(x, H, W);
    return {x, down, (H + 1) / 2, (W + 1) / 2};
  }
  return {x, x, H, W};
}

SwinTransformerImpl::SwinTransformerImpl(const SwinConfig& cfg) : out_indices_(cfg.out_indices) {
  const int num_layers = static_cast<int>(cfg.depths.size());
  patch_embed = register_module(
      "patch_embed", SwinPatchEmbed(cfg.patch_size, 3, cfg.embed_dim, /*norm=*/true));

  // Per-stage channel counts and which stages downsample (the last stage never does; the
  // dilation variant also drops the second-to-last downsample to keep a stride-16 output).
  num_features_.resize(static_cast<std::size_t>(num_layers));
  std::vector<bool> has_down(static_cast<std::size_t>(num_layers), true);
  for (int i = 0; i < num_layers; ++i) {
    num_features_[static_cast<std::size_t>(i)] = cfg.embed_dim * (1 << i);
  }
  has_down[static_cast<std::size_t>(num_layers - 1)] = false;
  if (cfg.dilation) {
    has_down[static_cast<std::size_t>(num_layers - 2)] = false;
    num_features_[static_cast<std::size_t>(num_layers - 1)] = (cfg.embed_dim * (1 << (num_layers - 1))) / 2;
  }

  layers = register_module("layers", nn::ModuleList());
  for (int i = 0; i < num_layers; ++i) {
    const auto u = static_cast<std::size_t>(i);
    layers->push_back(SwinBasicLayer(num_features_[u], cfg.depths[u], cfg.num_heads[u],
                                     cfg.window_size, cfg.mlp_ratio, has_down[u]));
  }

  // A LayerNorm per output stage, registered as "norm{i}" to match the checkpoint.
  out_norms_.reserve(out_indices_.size());
  for (int idx : out_indices_) {
    out_norms_.push_back(register_module(
        "norm" + std::to_string(idx),
        nn::LayerNorm(nn::LayerNormOptions({num_features_[static_cast<std::size_t>(idx)]}))));
  }
}

std::vector<torch::Tensor> SwinTransformerImpl::forward(torch::Tensor x) {
  auto feat = patch_embed->forward(x);  // [B, C, Wh, Ww]
  int Wh = static_cast<int>(feat.size(2)), Ww = static_cast<int>(feat.size(3));
  auto h = feat.flatten(2).transpose(1, 2);  // [B, Wh*Ww, C]

  std::vector<torch::Tensor> outs;
  for (std::size_t i = 0; i < layers->size(); ++i) {
    const int H = Wh, W = Ww;  // resolution feeding this stage / of its `out` map
    auto stage = layers[i]->as<SwinBasicLayerImpl>()->forward(h, H, W);
    h = stage.down;
    Wh = stage.Wh;
    Ww = stage.Ww;

    auto it = std::find(out_indices_.begin(), out_indices_.end(), static_cast<int>(i));
    if (it != out_indices_.end()) {
      const auto j = static_cast<std::size_t>(it - out_indices_.begin());
      auto x_out = out_norms_[j]->forward(stage.out);
      const int nf = num_features_[i];
      outs.push_back(x_out.view({-1, H, W, nf}).permute({0, 3, 1, 2}).contiguous());
    }
  }
  return outs;
}

std::vector<int> SwinTransformerImpl::out_channels() const {
  std::vector<int> ch;
  ch.reserve(out_indices_.size());
  for (int idx : out_indices_) ch.push_back(num_features_[static_cast<std::size_t>(idx)]);
  return ch;
}

SwinConfig SwinConfigFor(const std::string& name) {
  SwinConfig c;
  if (name == "swin_T_224_1k") {
    c = {96, {2, 2, 6, 2}, {3, 6, 12, 24}, 7, 4, 4.0, {0, 1, 2, 3}, false};
  } else if (name == "swin_S_224_1k") {
    c = {96, {2, 2, 18, 2}, {3, 6, 12, 24}, 7, 4, 4.0, {0, 1, 2, 3}, false};
  } else if (name == "swin_B_224_22k") {
    c = {128, {2, 2, 18, 2}, {4, 8, 16, 32}, 7, 4, 4.0, {0, 1, 2, 3}, false};
  } else if (name == "swin_B_384_22k") {
    c = {128, {2, 2, 18, 2}, {4, 8, 16, 32}, 12, 4, 4.0, {0, 1, 2, 3}, false};
  } else if (name == "swin_L_224_22k") {
    c = {192, {2, 2, 18, 2}, {6, 12, 24, 48}, 7, 4, 4.0, {0, 1, 2, 3}, false};
  } else if (name == "swin_L_384_22k") {
    c = {192, {2, 2, 18, 2}, {6, 12, 24, 48}, 12, 4, 4.0, {0, 1, 2, 3}, false};
  } else {
    throw std::invalid_argument("unknown Swin variant: " + name);
  }
  return c;
}

}  // namespace detr::models
