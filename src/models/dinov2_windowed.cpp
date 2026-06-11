// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/dinov2_windowed.hpp"

#include <cmath>
#include <cstdint>

namespace detr::models {

namespace {
namespace F = torch::nn::functional;
}  // namespace

Dinov2BlockImpl::Dinov2BlockImpl(int dim, int heads, int ffn, int num_windows)
    : heads_(heads), num_windows_(num_windows) {
  norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({dim}).eps(1e-6)));
  norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({dim}).eps(1e-6)));
  q = register_module("q", nn::Linear(dim, dim));
  k = register_module("k", nn::Linear(dim, dim));
  v = register_module("v", nn::Linear(dim, dim));
  proj = register_module("proj", nn::Linear(dim, dim));
  fc1 = register_module("fc1", nn::Linear(dim, ffn));
  fc2 = register_module("fc2", nn::Linear(ffn, dim));
  ls1 = register_parameter("ls1", torch::ones({dim}));
  ls2 = register_parameter("ls2", torch::ones({dim}));
}

torch::Tensor Dinov2BlockImpl::forward(torch::Tensor x, bool run_full) {
  const std::int64_t w2 = static_cast<std::int64_t>(num_windows_) * num_windows_;
  auto shortcut = x;
  auto h = x;
  if (run_full) {  // merge windows for cross-window (global) attention
    h = x.reshape({x.size(0) / w2, w2 * x.size(1), x.size(2)});
  }
  // pre-norm multi-head self-attention.
  auto n = norm1->forward(h);
  const auto b = n.size(0);
  const auto t = n.size(1);
  const auto c = n.size(2);
  const auto hd = c / heads_;
  auto reshape = [&](const torch::Tensor& y) {
    return y.view({b, t, heads_, hd}).transpose(1, 2);  // [b, heads, t, hd]
  };
  auto qh = reshape(q->forward(n));
  auto kh = reshape(k->forward(n));
  auto vh = reshape(v->forward(n));
  auto scores = torch::matmul(qh, kh.transpose(-1, -2)) / std::sqrt(static_cast<double>(hd));
  auto ctx = torch::matmul(scores.softmax(-1), vh);             // [b, heads, t, hd]
  ctx = ctx.transpose(1, 2).contiguous().view({b, t, c});
  auto attn = proj->forward(ctx);
  if (run_full) {  // split back into windows
    attn = attn.reshape({b * w2, t / w2, c});
  }
  attn = attn * ls1;
  auto hs = attn + shortcut;  // residual (windowed format)
  auto m = fc2->forward(torch::gelu(fc1->forward(norm2->forward(hs))));
  return m * ls2 + hs;
}

Dinov2WindowedImpl::Dinov2WindowedImpl(int embed, int depth, int heads, int patch, int num_windows,
                                       int pe_grid, int num_registers,
                                       std::vector<int> out_indices,
                                       std::vector<int> window_block_indexes)
    : embed_(embed),
      patch_(patch),
      num_windows_(num_windows),
      pe_grid_(pe_grid),
      num_registers_(num_registers),
      out_indices_(std::move(out_indices)),
      window_blocks_(window_block_indexes.begin(), window_block_indexes.end()) {
  patch_embed = register_module(
      "patch_embed", nn::Conv2d(nn::Conv2dOptions(3, embed, patch).stride(patch)));
  cls_token = register_parameter("cls_token", torch::zeros({1, 1, embed}));
  pos_embed = register_parameter("pos_embed", torch::zeros({1, pe_grid * pe_grid + 1, embed}));
  if (num_registers_ > 0) {
    reg_tokens = register_parameter("reg_tokens", torch::zeros({1, num_registers_, embed}));
  }
  blocks = register_module("blocks", nn::ModuleList());
  for (int i = 0; i < depth; ++i) {
    blocks->push_back(Dinov2Block(embed, heads, embed * 4, num_windows));
  }
  final_norm = register_module("final_norm", nn::LayerNorm(nn::LayerNormOptions({embed}).eps(1e-6)));
}

namespace {

// Position embedding for a grid of (h,w) patches; native grid returns it directly,
// otherwise bicubic-interpolates the patch part (cls kept separate).
torch::Tensor InterpPos(const torch::Tensor& pos_embed, int pe_grid, std::int64_t h,
                        std::int64_t w) {
  const auto npos = pos_embed.size(1) - 1;
  if (h * w == npos && h == w) {
    return pos_embed;
  }
  const auto dim = pos_embed.size(2);
  auto cls_pe = pos_embed.narrow(1, 0, 1);
  auto patch_pe = pos_embed.narrow(1, 1, npos)
                      .reshape({1, pe_grid, pe_grid, dim})
                      .permute({0, 3, 1, 2});  // [1,C,g,g]
  patch_pe = F::interpolate(patch_pe.to(torch::kFloat32),
                            F::InterpolateFuncOptions()
                                .size(std::vector<std::int64_t>{h, w})
                                .mode(torch::kBicubic)
                                .align_corners(false));
  patch_pe = patch_pe.permute({0, 2, 3, 1}).reshape({1, h * w, dim});
  return torch::cat({cls_pe, patch_pe}, 1);
}

}  // namespace

std::vector<torch::Tensor> Dinov2WindowedImpl::forward(torch::Tensor images) {
  const std::int64_t nw = num_windows_;
  auto p = patch_embed->forward(images);  // [B,C,h,w]
  const auto bsz = p.size(0);
  const auto c = p.size(1);
  const auto gh = p.size(2);
  const auto gw = p.size(3);
  auto emb = p.flatten(2).transpose(1, 2);                          // [B, h*w, C]
  emb = torch::cat({cls_token.expand({bsz, -1, -1}), emb}, 1);      // [B, h*w+1, C]
  emb = emb + InterpPos(pos_embed, pe_grid_, gh, gw);

  if (nw > 1) {  // window-partition the patch tokens; duplicate cls per window
    const auto hpw = gh / nw;
    const auto wpw = gw / nw;
    auto cls_pe = emb.narrow(1, 0, 1);
    auto pix = emb.narrow(1, 1, gh * gw).view({bsz, gh, gw, c});
    auto win = pix.reshape({bsz * nw, hpw, nw, wpw, c})
                   .permute({0, 2, 1, 3, 4})
                   .reshape({bsz * nw * nw, hpw * wpw, c});
    emb = torch::cat({cls_pe.repeat({nw * nw, 1, 1}), win}, 1);
  }
  if (num_registers_ > 0) {  // registers inserted right after cls
    auto cls_part = emb.narrow(1, 0, 1);
    auto rest = emb.narrow(1, 1, emb.size(1) - 1);
    emb = torch::cat({cls_part, reg_tokens.expand({emb.size(0), -1, -1}), rest}, 1);
  }

  // Un-window a block output [B*nw², T, C] -> [B, C, gh, gw].
  const auto skip = 1 + static_cast<std::int64_t>(num_registers_);
  auto unwindow = [&](const torch::Tensor& x) {
    if (nw <= 1) {
      return x.narrow(1, skip, gh * gw).transpose(1, 2).reshape({bsz, c, gh, gw});
    }
    const auto hpw = gh / nw;
    const auto wpw = gw / nw;
    auto pix = x.narrow(1, skip, hpw * wpw * 1);  // [B*nw², hpw*wpw, C]
    return pix.reshape({bsz, nw, nw, hpw, wpw, c})
        .permute({0, 1, 3, 2, 4, 5})
        .reshape({bsz, gh, gw, c})
        .permute({0, 3, 1, 2})
        .contiguous();
  };

  std::vector<torch::Tensor> feats;
  std::set<int> want;
  for (int s : out_indices_) {
    want.insert(s - 1);  // stage s == output of block (s-1)
  }
  const int max_block = out_indices_.back() - 1;
  auto x = emb;
  for (int i = 0; i <= max_block; ++i) {
    const bool run_full = window_blocks_.count(i) == 0;
    x = blocks[static_cast<std::size_t>(i)]->as<Dinov2BlockImpl>()->forward(x, run_full);
    if (want.count(i) != 0) {  // apply the final layernorm to the feature, then un-window
      feats.push_back(unwindow(final_norm->forward(x)));
    }
  }
  return feats;
}

}  // namespace detr::models
