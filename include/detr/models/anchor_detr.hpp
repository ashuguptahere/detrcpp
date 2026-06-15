// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Anchor-DETR (megvii-research/AnchorDETR): a DETR variant whose object queries are
// anchor points. A learned set of `num_position` 2D anchor points, each instantiated
// with `num_pattern` content patterns, replaces DETR's free query embeddings; the box
// head predicts an offset from the anchor. Its other novelty is Row-Column Decoupled
// Attention (RCDA), which factorizes the H*W cross/self-attention into a row (over W)
// and a column (over H) softmax whose outer product weights the value — O(L*(H+W))
// instead of O(L*H*W). Single feature level (C5 or DC5 ResNet), sigmoid-focal head.
// Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "detr/models/model.hpp"
#include "detr/models/resnet.hpp"

namespace detr::models {

namespace nn = torch::nn;

// Row-Column Decoupled Attention. One fused input projection (5*d: q_row, q_col, k_row,
// k_col, v) and an output projection. Queries are [B, L, d]; keys/value are spatial
// [B, H, W, d]. Returns [B, L, d].
struct RcdaImpl : nn::Module {
  RcdaImpl(int d_model, int nhead);
  torch::Tensor forward(const torch::Tensor& query_row, const torch::Tensor& query_col,
                        const torch::Tensor& key_row, const torch::Tensor& key_col,
                        const torch::Tensor& value);
  int nhead_, d_model_;
  torch::Tensor in_proj_weight, in_proj_bias;  // [5d, d], [5d]
  nn::Linear out_proj{nullptr};
};
TORCH_MODULE(Rcda);

// Position-embedding FFN: Linear(d,d) -> ReLU -> Linear(d,d) (the upstream adapt_pos*d).
struct AdaptPosImpl : nn::Module {
  explicit AdaptPosImpl(int d_model);
  torch::Tensor forward(const torch::Tensor& x);
  nn::Linear fc0{nullptr}, fc2{nullptr};
};
TORCH_MODULE(AdaptPos);

// A post-norm FFN block: Linear -> ReLU -> Linear, residual, LayerNorm.
struct AnchorFfnImpl : nn::Module {
  AnchorFfnImpl(int d_model, int d_ffn);
  torch::Tensor forward(torch::Tensor x);
  nn::Linear linear1{nullptr}, linear2{nullptr};
  nn::LayerNorm norm2{nullptr};
};
TORCH_MODULE(AnchorFfn);

// Spatial encoder layer: RCDA self-attention over the feature map + FFN.
struct AnchorEncoderLayerImpl : nn::Module {
  AnchorEncoderLayerImpl(int d_model, int d_ffn, int nhead);
  // src: [B, C, H, W]; posemb_row: [B, W, C]; posemb_col: [B, H, C]. Returns [B, C, H, W].
  torch::Tensor forward(torch::Tensor src, const torch::Tensor& posemb_row,
                        const torch::Tensor& posemb_col);
  Rcda self_attn{nullptr};
  nn::LayerNorm norm1{nullptr};
  AnchorFfn ffn{nullptr};
};
TORCH_MODULE(AnchorEncoderLayer);

// Decoder layer: standard MHA self-attention (anchor-point query pos) + RCDA cross-
// attention into the feature map + FFN.
struct AnchorDecoderLayerImpl : nn::Module {
  AnchorDecoderLayerImpl(int d_model, int d_ffn, int nhead);
  // tgt: [B, nq, C]; reference_points: [B, nq, 2]; srcs: [B, C, H, W].
  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& reference_points,
                        const torch::Tensor& srcs, const torch::Tensor& posemb_row,
                        const torch::Tensor& posemb_col, AdaptPos adapt_pos2d, AdaptPos adapt_pos1d);
  int nhead_;
  Rcda cross_attn{nullptr};
  nn::LayerNorm norm1{nullptr};
  nn::MultiheadAttention self_attn{nullptr};
  nn::LayerNorm norm2{nullptr};
  AnchorFfn ffn{nullptr};
};
TORCH_MODULE(AnchorDecoderLayer);

struct AnchorTransformerConfig {
  int d_model = 256, nhead = 8, enc_layers = 6, dec_layers = 6, dim_feedforward = 1024;
  int num_classes = 91, num_position = 300, num_pattern = 3;
};

// The Anchor-DETR transformer: anchor points + patterns -> queries, RCDA encoder, RCDA
// decoder, shared sigmoid-focal class head + 2D-refined box head per layer.
struct AnchorTransformerImpl : nn::Module {
  explicit AnchorTransformerImpl(const AnchorTransformerConfig& cfg);
  // src: [B, C, H, W]. Returns {logits [B, nq, num_classes], boxes [B, nq, 4] cxcywh}.
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& src);
  AnchorTransformerConfig cfg_;
  nn::Embedding pattern{nullptr}, position{nullptr};
  AdaptPos adapt_pos1d{nullptr}, adapt_pos2d{nullptr};
  nn::ModuleList encoder_layers{nullptr}, decoder_layers{nullptr};
  nn::Linear class_embed{nullptr};   // shared across decoder layers
  nn::ModuleList bbox_embed{nullptr};  // shared 3-layer MLP (registered as layers)
};
TORCH_MODULE(AnchorTransformer);

// Registers anchor-detr (C5) and anchor-detr-dc5 (dilated C5) into the global registry.
void RegisterAnchorDetr();

}  // namespace detr::models
