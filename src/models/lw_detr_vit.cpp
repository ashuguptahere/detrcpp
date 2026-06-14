// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/lw_detr_vit.hpp"

#include <cmath>
#include <cstdint>

namespace detr::models {

namespace {
namespace F = torch::nn::functional;
}  // namespace

LwDetrViTBlockImpl::LwDetrViTBlockImpl(int dim, int heads, int ffn, int num_windows_side)
    : heads_(heads), nws_(num_windows_side) {
  norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({dim}).eps(1e-6)));
  norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({dim}).eps(1e-6)));
  q = register_module("q", nn::Linear(nn::LinearOptions(dim, dim)));
  k = register_module("k", nn::Linear(nn::LinearOptions(dim, dim).bias(false)));  // k_proj: no bias
  v = register_module("v", nn::Linear(nn::LinearOptions(dim, dim)));
  o = register_module("o", nn::Linear(nn::LinearOptions(dim, dim)));
  fc1 = register_module("fc1", nn::Linear(dim, ffn));
  fc2 = register_module("fc2", nn::Linear(ffn, dim));
  gamma1 = register_parameter("gamma1", torch::ones({dim}));
  gamma2 = register_parameter("gamma2", torch::ones({dim}));
}

torch::Tensor LwDetrViTBlockImpl::forward(torch::Tensor x, bool run_full) {
  const std::int64_t w2 = static_cast<std::int64_t>(nws_) * nws_;
  auto shortcut = x;  // windowed format [B*w2, t, c]
  auto h = run_full ? x.reshape({x.size(0) / w2, w2 * x.size(1), x.size(2)}) : x;

  auto n = norm1->forward(h);
  const auto b = n.size(0);
  const auto t = n.size(1);
  const auto c = n.size(2);
  const auto hd = c / heads_;
  auto split = [&](const torch::Tensor& y) { return y.view({b, t, heads_, hd}).transpose(1, 2); };
  auto qh = split(q->forward(n));
  auto kh = split(k->forward(n));
  auto vh = split(v->forward(n));
  auto scores = torch::matmul(qh, kh.transpose(-1, -2)) / std::sqrt(static_cast<double>(hd));
  auto ctx = torch::matmul(scores.softmax(-1), vh).transpose(1, 2).contiguous().view({b, t, c});
  auto attn = o->forward(ctx) * gamma1;
  if (run_full) {  // back to windowed format
    attn = attn.reshape({b * w2, t / w2, c});
  }
  auto hs = shortcut + attn;
  auto m = fc2->forward(torch::gelu(fc1->forward(norm2->forward(hs)))) * gamma2;
  return hs + m;
}

LwDetrViTImpl::LwDetrViTImpl(int embed, int depth, int heads, int patch, int num_windows_side,
                             int pe_grid, std::vector<int> out_layer_indexes,
                             std::vector<int> window_block_indexes)
    : embed_(embed),
      patch_(patch),
      nws_(num_windows_side),
      pe_grid_(pe_grid),
      out_layers_(std::move(out_layer_indexes)),
      window_blocks_(window_block_indexes.begin(), window_block_indexes.end()) {
  patch_embed =
      register_module("patch_embed", nn::Conv2d(nn::Conv2dOptions(3, embed, patch).stride(patch)));
  pos_embed = register_parameter("pos_embed", torch::zeros({1, pe_grid * pe_grid + 1, embed}));
  blocks = register_module("blocks", nn::ModuleList());
  for (int i = 0; i < depth; ++i) {
    blocks->push_back(LwDetrViTBlock(embed, heads, embed * 4, num_windows_side));
  }
}

namespace {

// Absolute pos-embed for a (h,w) patch grid: drop the cls slot, bicubic-interpolate
// the square patch grid to (h,w). Returns [1, h, w, C] (added in channels-last layout).
torch::Tensor InterpPos(const torch::Tensor& pos_embed, int pe_grid, std::int64_t h,
                        std::int64_t w) {
  const auto npos = pos_embed.size(1) - 1;
  const auto dim = pos_embed.size(2);
  auto patch_pe = pos_embed.narrow(1, 1, npos)
                      .reshape({1, pe_grid, pe_grid, dim})
                      .permute({0, 3, 1, 2});  // [1,C,g,g]
  if (h != pe_grid || w != pe_grid) {
    patch_pe = F::interpolate(patch_pe.to(torch::kFloat32),
                              F::InterpolateFuncOptions()
                                  .size(std::vector<std::int64_t>{h, w})
                                  .mode(torch::kBicubic)
                                  .align_corners(false));
  }
  return patch_pe.permute({0, 2, 3, 1}).contiguous();  // [1,h,w,C]
}

}  // namespace

std::vector<torch::Tensor> LwDetrViTImpl::forward(torch::Tensor images) {
  const std::int64_t nws = nws_;
  auto p = patch_embed->forward(images);  // [B,C,H,W]
  const auto bsz = p.size(0);
  const auto c = p.size(1);
  const auto gh = p.size(2);
  const auto gw = p.size(3);

  // [B,C,H,W] -> [B,H,W,C], add absolute pos-embed.
  auto hwc = p.permute({0, 2, 3, 1}) + InterpPos(pos_embed, pe_grid_, gh, gw);

  // Window-partition: [B,H,W,C] -> [B*nws^2, (H/nws)*(W/nws), C].
  const auto wh = gh / nws;
  const auto ww = gw / nws;
  auto x = hwc.reshape({bsz, nws, wh, nws, ww, c})
               .permute({0, 1, 3, 2, 4, 5})
               .reshape({bsz * nws * nws, wh * ww, c});

  // Un-window a block output [B*nws^2, wh*ww, C] -> [B, C, H, W].
  auto unwindow = [&](const torch::Tensor& t) {
    return t.reshape({bsz, nws, nws, wh, ww, c})
        .permute({0, 5, 1, 3, 2, 4})
        .reshape({bsz, c, gh, gw})
        .contiguous();
  };

  std::set<int> want(out_layers_.begin(), out_layers_.end());
  const int max_block = out_layers_.back();
  std::vector<torch::Tensor> feats;
  for (int i = 0; i <= max_block; ++i) {
    const bool run_full = window_blocks_.count(i) == 0;  // global block merges windows
    x = blocks[static_cast<std::size_t>(i)]->as<LwDetrViTBlockImpl>()->forward(x, run_full);
    if (want.count(i) != 0) {
      feats.push_back(unwindow(x));
    }
  }
  return feats;
}

}  // namespace detr::models
