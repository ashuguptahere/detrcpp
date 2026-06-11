// Copyright 2026 detrcpp authors. Apache-2.0.
//
// The decoupled content/spatial cross-attention decoder layer shared by
// Conditional-DETR and DAB-DETR: standard self-attention (externally projected)
// plus a cross-attention that concatenates content and spatial queries along the
// head dim (Q/K are 2x width, V is width). Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

namespace detr::models {

// Multi-head attention with externally-projected q/k/v (no in_proj), allowing q/k
// of one width and v of another. Inputs [L, B, dim]; returns [Lq, B, v_dim].
// |attn_mask| (optional [Lq,Lk] bool, true = block) masks scores before softmax.
torch::Tensor DecoupledMultiHeadAttn(torch::Tensor q, torch::Tensor k, torch::Tensor v, int nhead,
                                     const torch::Tensor& attn_mask = {});

struct CondDecoderLayerImpl : torch::nn::Module {
  torch::nn::Linear sa_qcontent{nullptr}, sa_qpos{nullptr}, sa_kcontent{nullptr}, sa_kpos{nullptr},
      sa_v{nullptr}, sa_out{nullptr};
  torch::nn::Linear ca_qcontent{nullptr}, ca_qpos{nullptr}, ca_kcontent{nullptr}, ca_kpos{nullptr},
      ca_v{nullptr}, ca_qpos_sine{nullptr}, ca_out{nullptr};
  torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
  torch::nn::Linear linear1{nullptr}, linear2{nullptr};
  // FFN activation: relu by default (Conditional-DETR), or a learnable PReLU
  // (DAB-DETR, registered as "activation_fn" so the official weight loads 1:1).
  torch::nn::PReLU activation_fn{nullptr};
  bool use_prelu_{false};
  int nhead_;
  int d_;

  CondDecoderLayerImpl(int d, int nhead, int ff, bool use_prelu = false);

  // |query_sine| is the (already projected/modulated) spatial query; |is_first|
  // adds the query positional embedding to content on the first layer only.
  // |self_attn_mask| (optional [L,L] bool, true = block) masks self-attention only
  // (DN-DETR group isolation); the cross-attention is always unmasked.
  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& memory, const torch::Tensor& pos,
                        const torch::Tensor& query_pos, const torch::Tensor& query_sine,
                        bool is_first, const torch::Tensor& self_attn_mask = {});
};
TORCH_MODULE(CondDecoderLayer);

}  // namespace detr::models
