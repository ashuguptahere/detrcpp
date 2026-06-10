// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/cond_decoder.hpp"

#include <cmath>
#include <cstdint>

namespace detr::models {

namespace nn = torch::nn;

torch::Tensor DecoupledMultiHeadAttn(torch::Tensor q, torch::Tensor k, torch::Tensor v, int nhead) {
  const auto lq = q.size(0);
  const auto b = q.size(1);
  const auto lk = k.size(0);
  const auto qk_hd = q.size(2) / nhead;
  const auto v_hd = v.size(2) / nhead;
  auto split = [&](torch::Tensor t, std::int64_t l, std::int64_t hd) {
    return t.view({l, b, nhead, hd}).permute({1, 2, 0, 3}).reshape({b * nhead, l, hd});
  };
  auto qh = split(q, lq, qk_hd);
  auto kh = split(k, lk, qk_hd);
  auto vh = split(v, lk, v_hd);
  auto scores = torch::bmm(qh, kh.transpose(1, 2)) / std::sqrt(static_cast<double>(qk_hd));
  auto out = torch::bmm(scores.softmax(-1), vh);  // [b*nh, lq, v_hd]
  return out.view({b, nhead, lq, v_hd}).permute({2, 0, 1, 3}).reshape({lq, b, nhead * v_hd});
}

CondDecoderLayerImpl::CondDecoderLayerImpl(int d, int nhead, int ff, bool use_prelu)
    : use_prelu_(use_prelu), nhead_(nhead), d_(d) {
  auto lin = [&](const char* n) { return register_module(n, nn::Linear(d, d)); };
  sa_qcontent = lin("sa_qcontent_proj");
  sa_qpos = lin("sa_qpos_proj");
  sa_kcontent = lin("sa_kcontent_proj");
  sa_kpos = lin("sa_kpos_proj");
  sa_v = lin("sa_v_proj");
  sa_out = lin("sa_out_proj");
  ca_qcontent = lin("ca_qcontent_proj");
  ca_qpos = lin("ca_qpos_proj");
  ca_kcontent = lin("ca_kcontent_proj");
  ca_kpos = lin("ca_kpos_proj");
  ca_v = lin("ca_v_proj");
  ca_qpos_sine = lin("ca_qpos_sine_proj");
  ca_out = lin("ca_out_proj");
  norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
  norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
  norm3 = register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({d})));
  linear1 = register_module("linear1", nn::Linear(d, ff));
  linear2 = register_module("linear2", nn::Linear(ff, d));
  if (use_prelu_) {
    activation_fn = register_module("activation_fn", nn::PReLU());  // num_parameters=1
  }
}

torch::Tensor CondDecoderLayerImpl::forward(torch::Tensor tgt, const torch::Tensor& memory,
                                            const torch::Tensor& pos,
                                            const torch::Tensor& query_pos,
                                            const torch::Tensor& query_sine, bool is_first) {
  const auto nq = tgt.size(0);
  const auto b = tgt.size(1);
  const auto hw = memory.size(0);
  const auto hd = d_ / nhead_;

  // self-attention (q/k/v projected externally; no in_proj).
  auto q = sa_qcontent->forward(tgt) + sa_qpos->forward(query_pos);
  auto k = sa_kcontent->forward(tgt) + sa_kpos->forward(query_pos);
  auto sa = sa_out->forward(DecoupledMultiHeadAttn(q, k, sa_v->forward(tgt), nhead_));
  tgt = norm1->forward(tgt + sa);

  // conditional cross-attention.
  auto qc = ca_qcontent->forward(tgt);
  auto kc = ca_kcontent->forward(memory);
  auto v = ca_v->forward(memory);
  auto kp = ca_kpos->forward(pos);
  torch::Tensor qq;
  torch::Tensor kk;
  if (is_first) {
    qq = qc + ca_qpos->forward(query_pos);
    kk = kc + kp;
  } else {
    qq = qc;
    kk = kc;
  }
  auto qse = ca_qpos_sine->forward(query_sine).view({nq, b, nhead_, hd});
  auto qcat = torch::cat({qq.view({nq, b, nhead_, hd}), qse}, 3).view({nq, b, 2 * d_});
  auto kcat = torch::cat({kk.view({hw, b, nhead_, hd}), kp.view({hw, b, nhead_, hd})}, 3)
                  .view({hw, b, 2 * d_});
  auto ca = ca_out->forward(DecoupledMultiHeadAttn(qcat, kcat, v, nhead_));
  tgt = norm2->forward(tgt + ca);

  auto hidden = linear1->forward(tgt);
  hidden = use_prelu_ ? activation_fn->forward(hidden) : torch::relu(hidden);
  auto ff = linear2->forward(hidden);
  return norm3->forward(tgt + ff);
}

}  // namespace detr::models
