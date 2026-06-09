// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/vit.hpp"

#include <cmath>
#include <cstdint>

namespace detr::models {

namespace {
namespace nn = torch::nn;

// 2D sine position for the patch grid: [1, H*W, dim] (parameter-free, so the
// backbone handles any input size).
torch::Tensor SinCos2d(std::int64_t h, std::int64_t w, int dim, const torch::TensorOptions& opts) {
  auto gh = torch::arange(h, opts);
  auto gw = torch::arange(w, opts);
  auto grid = torch::meshgrid({gh, gw}, "ij");  // [h,w]
  const int pos_dim = dim / 4;
  auto omega = torch::arange(pos_dim, opts) / static_cast<double>(pos_dim);
  omega = 1.0 / torch::pow(10000.0, omega);
  auto oh = grid[0].reshape(-1).unsqueeze(1) * omega.unsqueeze(0);
  auto ow = grid[1].reshape(-1).unsqueeze(1) * omega.unsqueeze(0);
  return torch::cat({oh.sin(), oh.cos(), ow.sin(), ow.cos()}, 1).unsqueeze(0);  // [1, h*w, dim]
}

// Pre-norm transformer block.
struct BlockImpl : nn::Module {
  nn::LayerNorm norm1{nullptr};
  nn::LayerNorm norm2{nullptr};
  nn::MultiheadAttention attn{nullptr};
  nn::Linear fc1{nullptr};
  nn::Linear fc2{nullptr};
  BlockImpl(int dim, int nheads, int mlp_ratio) {
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({dim})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({dim})));
    attn =
        register_module("attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(dim, nheads)));
    fc1 = register_module("fc1", nn::Linear(dim, dim * mlp_ratio));
    fc2 = register_module("fc2", nn::Linear(dim * mlp_ratio, dim));
  }
  torch::Tensor forward(torch::Tensor x, const torch::Tensor& pos) {
    auto n = norm1->forward(x);
    auto q = (n + pos).transpose(0, 1);  // [L, B, dim]
    auto a = std::get<0>(attn->forward(q, q, n.transpose(0, 1))).transpose(0, 1);
    x = x + a;
    x = x + fc2->forward(torch::gelu(fc1->forward(norm2->forward(x))));
    return x;
  }
};
TORCH_MODULE(Block);

}  // namespace

ViTImpl::ViTImpl(int embed_dim, int depth, int nheads, int patch, int mlp_ratio)
    : embed_dim_(embed_dim), patch_(patch), nheads_(nheads) {
  patch_embed = register_module("patch_embed",
                                nn::Conv2d(nn::Conv2dOptions(3, embed_dim, patch).stride(patch)));
  blocks = register_module("blocks", nn::ModuleList());
  for (int i = 0; i < depth; ++i) {
    blocks->push_back(Block(embed_dim, nheads, mlp_ratio));
  }
  norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({embed_dim})));
}

torch::Tensor ViTImpl::forward(torch::Tensor x) {
  x = patch_embed->forward(x);  // [B, dim, h, w]
  const auto b = x.size(0);
  const auto dim = x.size(1);
  const auto h = x.size(2);
  const auto w = x.size(3);
  auto pos = SinCos2d(h, w, static_cast<int>(dim), x.options());  // [1, h*w, dim]
  auto tokens = x.flatten(2).transpose(1, 2);                     // [B, h*w, dim]
  for (const auto& m : *blocks) {
    tokens = m->as<BlockImpl>()->forward(tokens, pos);
  }
  tokens = norm->forward(tokens);
  return tokens.transpose(1, 2).reshape({b, dim, h, w});  // [B, dim, h, w]
}

}  // namespace detr::models
