// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/dinov3_sta.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace detr::models {

namespace {

namespace F = torch::nn::functional;

// Splits the last dim in half and swaps with a sign flip: [x1, x2] -> [-x2, x1].
torch::Tensor RotateHalf(const torch::Tensor& x) {
  const auto half = x.size(-1) / 2;
  auto x1 = x.slice(-1, 0, half);
  auto x2 = x.slice(-1, half, x.size(-1));
  return torch::cat({-x2, x1}, -1);
}

// Rotary embedding: x * cos + rotate_half(x) * sin.
torch::Tensor ApplyRope(const torch::Tensor& x, const torch::Tensor& sin, const torch::Tensor& cos) {
  return x * cos + RotateHalf(x) * sin;
}

}  // namespace

RopeEmbedImpl::RopeEmbedImpl(int head_dim) {
  periods = register_buffer("periods", torch::empty({head_dim / 4}));
}

std::pair<torch::Tensor, torch::Tensor> RopeEmbedImpl::SinCos(int h, int w) const {
  const auto p = periods.size(0);
  const auto opts = periods.options();
  auto coords_h = torch::arange(0.5, static_cast<double>(h), 1.0, opts) / h;  // [h]
  auto coords_w = torch::arange(0.5, static_cast<double>(w), 1.0, opts) / w;  // [w]
  auto hh = coords_h.view({h, 1}).expand({h, w});
  auto ww = coords_w.view({1, w}).expand({h, w});
  auto coords = torch::stack({hh, ww}, -1).reshape({h * w, 2});  // [N, 2]
  coords = 2.0 * coords - 1.0;
  auto angles = (2.0 * M_PI) * coords.unsqueeze(-1) / periods.view({1, 1, p});  // [N, 2, P]
  angles = angles.flatten(1, 2).repeat({1, 2});                                 // [N, head_dim]
  auto sin = angles.sin().unsqueeze(0).unsqueeze(0);                            // [1, 1, N, head_dim]
  auto cos = angles.cos().unsqueeze(0).unsqueeze(0);
  return {sin, cos};
}

VitRopeBlockImpl::VitRopeBlockImpl(int dim, int num_heads, double mlp_ratio)
    : num_heads_(num_heads), scale_(std::pow(static_cast<double>(dim / num_heads), -0.5)) {
  norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({dim}).eps(1e-6)));
  norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({dim}).eps(1e-6)));
  auto attn = register_module("attn", std::make_shared<nn::Module>());
  qkv = attn->register_module("qkv", nn::Linear(dim, dim * 3));
  proj = attn->register_module("proj", nn::Linear(dim, dim));
  auto mlp = register_module("mlp", std::make_shared<nn::Module>());
  const int hidden = static_cast<int>(dim * mlp_ratio);
  fc1 = mlp->register_module("fc1", nn::Linear(dim, hidden));
  fc2 = mlp->register_module("fc2", nn::Linear(hidden, dim));
}

torch::Tensor VitRopeBlockImpl::forward(torch::Tensor x, const torch::Tensor& sin,
                                        const torch::Tensor& cos) {
  // Self-attention (RoPE on the patch tokens; cls token at index 0 is left as-is).
  auto y = norm1->forward(x);
  const auto B = y.size(0), N = y.size(1), C = y.size(2);
  auto qkv_out = qkv->forward(y).reshape({B, N, 3, num_heads_, C / num_heads_}).permute({2, 0, 3, 1, 4});
  auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2];  // each [B, heads, N, head_dim]
  auto qp = ApplyRope(q.slice(2, 1, N), sin, cos);
  auto kp = ApplyRope(k.slice(2, 1, N), sin, cos);
  q = torch::cat({q.slice(2, 0, 1), qp}, 2);
  k = torch::cat({k.slice(2, 0, 1), kp}, 2);
  auto attn = torch::softmax(torch::matmul(q, k.transpose(-2, -1)) * scale_, -1);
  auto o = torch::matmul(attn, v).transpose(1, 2).reshape({B, N, C});
  x = x + proj->forward(o);
  // Pre-norm GELU MLP.
  x = x + fc2->forward(torch::gelu(fc1->forward(norm2->forward(x))));
  return x;
}

VitRopeModelImpl::VitRopeModelImpl(int embed_dim, int num_heads, int depth, int patch_size,
                                   double mlp_ratio) {
  auto pe = register_module("patch_embed", std::make_shared<nn::Module>());
  patch_embed_proj = pe->register_module(
      "proj", nn::Conv2d(nn::Conv2dOptions(3, embed_dim, patch_size).stride(patch_size)));
  cls_token = register_parameter("cls_token", torch::zeros({1, 1, embed_dim}));
  blocks = register_module("blocks", nn::ModuleList());
  for (int i = 0; i < depth; ++i) blocks->push_back(VitRopeBlock(embed_dim, num_heads, mlp_ratio));
  rope_embed = register_module("rope_embed", RopeEmbed(embed_dim / num_heads));
}

VitRopeImpl::VitRopeImpl(int embed_dim, int num_heads, int depth, int patch_size,
                         std::vector<int> return_layers, double mlp_ratio)
    : embed_dim_(embed_dim), patch_size_(patch_size), return_layers_(std::move(return_layers)) {
  model = register_module("_model", VitRopeModel(embed_dim, num_heads, depth, patch_size, mlp_ratio));
}

std::vector<torch::Tensor> VitRopeImpl::forward(torch::Tensor x) {
  const auto B = x.size(0);
  const int gh = static_cast<int>(x.size(2)) / patch_size_;
  const int gw = static_cast<int>(x.size(3)) / patch_size_;
  auto tokens = model->patch_embed_proj->forward(x).flatten(2).transpose(1, 2);  // [B, N, C]
  auto cls = model->cls_token.expand({B, -1, -1});
  auto h = torch::cat({cls, tokens}, 1);  // [B, N+1, C]
  auto [sin, cos] = model->rope_embed->SinCos(gh, gw);
  std::vector<torch::Tensor> outs;
  for (std::size_t i = 0; i < model->blocks->size(); ++i) {
    h = model->blocks[i]->as<VitRopeBlockImpl>()->forward(h, sin, cos);
    if (std::find(return_layers_.begin(), return_layers_.end(), static_cast<int>(i)) !=
        return_layers_.end()) {
      outs.push_back(h.slice(1, 1, h.size(1)));  // patch tokens only (drop cls)
    }
  }
  return outs;
}

namespace {
// 3x3 stride-2 conv (no bias) used throughout the SpatialPriorModule.
nn::Conv2d Spm3x3(int in_ch, int out_ch) {
  return nn::Conv2d(nn::Conv2dOptions(in_ch, out_ch, 3).stride(2).padding(1).bias(false));
}
}  // namespace

SpatialPriorImpl::SpatialPriorImpl(int inplanes) {
  stem = register_module("stem", nn::Sequential());
  stem->push_back(Spm3x3(3, inplanes));
  stem->push_back(nn::BatchNorm2d(inplanes));
  stem->push_back(nn::GELU());
  stem->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2).padding(1)));
  conv2 = register_module("conv2", nn::Sequential());
  conv2->push_back(Spm3x3(inplanes, 2 * inplanes));
  conv2->push_back(nn::BatchNorm2d(2 * inplanes));
  conv3 = register_module("conv3", nn::Sequential());
  conv3->push_back(nn::GELU());
  conv3->push_back(Spm3x3(2 * inplanes, 4 * inplanes));
  conv3->push_back(nn::BatchNorm2d(4 * inplanes));
  conv4 = register_module("conv4", nn::Sequential());
  conv4->push_back(nn::GELU());
  conv4->push_back(Spm3x3(4 * inplanes, 4 * inplanes));
  conv4->push_back(nn::BatchNorm2d(4 * inplanes));
}

std::vector<torch::Tensor> SpatialPriorImpl::forward(torch::Tensor x) {
  auto c1 = stem->forward(x);     // 1/4
  auto c2 = conv2->forward(c1);   // 1/8
  auto c3 = conv3->forward(c2);   // 1/16
  auto c4 = conv4->forward(c3);   // 1/32
  return {c2, c3, c4};
}

DinoLayerScaleImpl::DinoLayerScaleImpl(int dim) {
  gamma = register_parameter("gamma", torch::ones({dim}));
}

DinoV3BlockImpl::DinoV3BlockImpl(int dim, int num_heads, double ffn_ratio, bool swiglu)
    : num_heads_(num_heads),
      scale_(std::pow(static_cast<double>(dim / num_heads), -0.5)),
      swiglu_(swiglu) {
  norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({dim}).eps(1e-5)));
  norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({dim}).eps(1e-5)));
  auto attn = register_module("attn", std::make_shared<nn::Module>());
  qkv = attn->register_module("qkv", nn::Linear(dim, dim * 3));
  proj = attn->register_module("proj", nn::Linear(dim, dim));
  ls1 = register_module("ls1", DinoLayerScale(dim));
  ls2 = register_module("ls2", DinoLayerScale(dim));
  auto mlp = register_module("mlp", std::make_shared<nn::Module>());
  const int hidden = static_cast<int>(dim * ffn_ratio);
  if (swiglu_) {
    const int d = static_cast<int>(hidden * 2.0 / 3.0);
    const int sw = (d + 7) / 8 * 8;  // round up to a multiple of 8 (align_to)
    w1 = mlp->register_module("w1", nn::Linear(dim, sw));
    w2 = mlp->register_module("w2", nn::Linear(dim, sw));
    w3 = mlp->register_module("w3", nn::Linear(sw, dim));
  } else {
    fc1 = mlp->register_module("fc1", nn::Linear(dim, hidden));
    fc2 = mlp->register_module("fc2", nn::Linear(hidden, dim));
  }
}

torch::Tensor DinoV3BlockImpl::forward(torch::Tensor x, const torch::Tensor& sin,
                                       const torch::Tensor& cos, int prefix) {
  auto y = norm1->forward(x);
  const auto B = y.size(0), N = y.size(1), C = y.size(2);
  auto qkv_out = qkv->forward(y).reshape({B, N, 3, num_heads_, C / num_heads_}).permute({2, 0, 3, 1, 4});
  auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2];
  auto qp = ApplyRope(q.slice(2, prefix, N), sin, cos);
  auto kp = ApplyRope(k.slice(2, prefix, N), sin, cos);
  q = torch::cat({q.slice(2, 0, prefix), qp}, 2);
  k = torch::cat({k.slice(2, 0, prefix), kp}, 2);
  auto attn = torch::softmax(torch::matmul(q, k.transpose(-2, -1)) * scale_, -1);
  auto o = torch::matmul(attn, v).transpose(1, 2).reshape({B, N, C});
  x = x + ls1->forward(proj->forward(o));
  auto n2 = norm2->forward(x);
  auto ffn = swiglu_ ? w3->forward(torch::silu(w1->forward(n2)) * w2->forward(n2))
                     : fc2->forward(torch::gelu(fc1->forward(n2)));
  x = x + ls2->forward(ffn);
  return x;
}

DinoV3VitImpl::DinoV3VitImpl(int embed_dim, int num_heads, int depth, int patch_size,
                             std::vector<int> return_layers, double ffn_ratio, bool swiglu,
                             int n_storage)
    : embed_dim_(embed_dim),
      patch_size_(patch_size),
      n_storage_(n_storage),
      return_layers_(std::move(return_layers)) {
  cls_token = register_parameter("cls_token", torch::zeros({1, 1, embed_dim}));
  storage_tokens = register_parameter("storage_tokens", torch::zeros({1, n_storage, embed_dim}));
  auto pe = register_module("patch_embed", std::make_shared<nn::Module>());
  patch_embed_proj = pe->register_module(
      "proj", nn::Conv2d(nn::Conv2dOptions(3, embed_dim, patch_size).stride(patch_size)));
  blocks = register_module("blocks", nn::ModuleList());
  for (int i = 0; i < depth; ++i) blocks->push_back(DinoV3Block(embed_dim, num_heads, ffn_ratio, swiglu));
  rope_embed = register_module("rope_embed", RopeEmbed(embed_dim / num_heads));
  norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({embed_dim}).eps(1e-5)));
}

std::vector<torch::Tensor> DinoV3VitImpl::forward(torch::Tensor x) {
  const auto B = x.size(0);
  const int gh = static_cast<int>(x.size(2)) / patch_size_;
  const int gw = static_cast<int>(x.size(3)) / patch_size_;
  auto tokens = patch_embed_proj->forward(x).flatten(2).transpose(1, 2);  // [B, HW, C]
  auto cls = cls_token.expand({B, -1, -1});
  auto stor = storage_tokens.expand({B, -1, -1});
  auto h = torch::cat({cls, stor, tokens}, 1);  // [B, 1 + n_storage + HW, C]
  const int prefix = 1 + n_storage_;
  auto [sin, cos] = rope_embed->SinCos(gh, gw);
  std::vector<torch::Tensor> outs;
  for (std::size_t i = 0; i < blocks->size(); ++i) {
    h = blocks[i]->as<DinoV3BlockImpl>()->forward(h, sin, cos, prefix);
    if (std::find(return_layers_.begin(), return_layers_.end(), static_cast<int>(i)) !=
        return_layers_.end()) {
      auto normed = norm->forward(h);  // final norm applied to each intermediate
      outs.push_back(normed.slice(1, prefix, normed.size(1)));  // patch tokens only
    }
  }
  return outs;
}

DinoV3StaImpl::DinoV3StaImpl(const DinoStaConfig& cfg)
    : patch_size_(cfg.patch_size), hidden_dim_(cfg.hidden_dim) {
  if (cfg.dinov3_vit) {
    dinov3_vit = register_module(
        "dinov3", DinoV3Vit(cfg.embed_dim, cfg.num_heads, cfg.depth, cfg.patch_size,
                            cfg.interaction_indexes, cfg.ffn_ratio, cfg.swiglu, cfg.n_storage));
  } else {
    dinov3 = register_module("dinov3", VitRope(cfg.embed_dim, cfg.num_heads, cfg.depth,
                                               cfg.patch_size, cfg.interaction_indexes));
  }
  sta = register_module("sta", SpatialPrior(cfg.conv_inplane));
  convs = register_module("convs", nn::ModuleList());
  // Level 0 fuses 2*inplanes detail channels, levels 1/2 fuse 4*inplanes.
  for (int extra : {2 * cfg.conv_inplane, 4 * cfg.conv_inplane, 4 * cfg.conv_inplane}) {
    convs->push_back(
        nn::Conv2d(nn::Conv2dOptions(cfg.embed_dim + extra, cfg.hidden_dim, 1).bias(false)));
  }
  norms = register_module("norms", nn::ModuleList());
  for (int i = 0; i < 3; ++i) norms->push_back(nn::BatchNorm2d(cfg.hidden_dim));
}

std::vector<torch::Tensor> DinoV3StaImpl::forward(torch::Tensor x) {
  const auto B = x.size(0);
  const int h_c = static_cast<int>(x.size(2)) / 16;  // the ViT token grid is always 1/16
  const int w_c = static_cast<int>(x.size(3)) / 16;
  auto layers = dinov3_vit ? dinov3_vit->forward(x) : dinov3->forward(x);  // 3 x [B, N, C]
  const int num_scales = static_cast<int>(layers.size()) - 2;
  std::vector<torch::Tensor> sem;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    auto feat = layers[i].transpose(1, 2).reshape({B, -1, h_c, w_c}).contiguous();  // [B, C, H, W]
    const double sc = std::pow(2.0, static_cast<double>(num_scales) - static_cast<double>(i));
    const auto rh = static_cast<std::int64_t>(h_c * sc), rw = static_cast<std::int64_t>(w_c * sc);
    sem.push_back(F::interpolate(
        feat, F::InterpolateFuncOptions().size(std::vector<std::int64_t>{rh, rw}).mode(torch::kBilinear).align_corners(false)));
  }
  auto detail = sta->forward(x);  // {c2, c3, c4}
  std::vector<torch::Tensor> out;
  for (std::size_t i = 0; i < 3; ++i) {
    auto fused = torch::cat({sem[i], detail[i]}, 1);
    auto c = convs[i]->as<nn::Conv2dImpl>()->forward(fused);
    out.push_back(norms[i]->as<nn::BatchNorm2dImpl>()->forward(c));
  }
  return out;
}

}  // namespace detr::models
