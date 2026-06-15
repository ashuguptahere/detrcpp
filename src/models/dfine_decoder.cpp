// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/dfine_decoder.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace detr::models {

namespace {

namespace F = torch::nn::functional;

// Non-uniform weighting function W(n): [reg_max+1] bin weights, exponentially spaced
// (fine near the centre, coarse at the ends). project for the distribution integral.
torch::Tensor WeightingFunction(int reg_max, double up, double reg_scale) {
  const double ub1 = std::abs(up) * std::abs(reg_scale);
  const double ub2 = ub1 * 2.0;
  const double step = std::pow(ub1 + 1.0, 2.0 / (reg_max - 2));
  std::vector<float> v;
  v.push_back(static_cast<float>(-ub2));
  for (int i = reg_max / 2 - 1; i >= 1; --i) v.push_back(static_cast<float>(-std::pow(step, i) + 1.0));
  v.push_back(0.0F);
  for (int i = 1; i < reg_max / 2; ++i) v.push_back(static_cast<float>(std::pow(step, i) - 1.0));
  v.push_back(static_cast<float>(ub2));
  return torch::tensor(v);
}

// Grid-anchor priors per level (unsigmoid), with a validity mask (centre/size inside eps).
std::pair<torch::Tensor, torch::Tensor> GenerateAnchors(const DfShapes& shapes, double eps,
                                                        const torch::TensorOptions& opts) {
  std::vector<torch::Tensor> anchors;
  for (std::size_t lvl = 0; lvl < shapes.size(); ++lvl) {
    const auto h = shapes[lvl].first, w = shapes[lvl].second;
    auto grid = torch::meshgrid({torch::arange(h, opts), torch::arange(w, opts)}, "ij");
    auto grid_xy = torch::stack({grid[1], grid[0]}, -1);  // (x, y)
    grid_xy = (grid_xy.unsqueeze(0) + 0.5) /
              torch::tensor({static_cast<float>(w), static_cast<float>(h)}, opts);
    auto wh = torch::ones_like(grid_xy) * 0.05 * std::pow(2.0, static_cast<double>(lvl));
    anchors.push_back(torch::cat({grid_xy, wh}, -1).reshape({1, h * w, 4}));
  }
  auto a = torch::cat(anchors, 1);
  auto valid = ((a > eps) & (a < 1.0 - eps)).all(-1, /*keepdim=*/true);
  a = torch::log(a / (1.0 - a));
  a = torch::where(valid, a, torch::full_like(a, std::numeric_limits<float>::infinity()));
  return {a, valid};
}

// Decode integrated edge distances + anchor box -> cxcywh box.
torch::Tensor Distance2Bbox(const torch::Tensor& points, const torch::Tensor& distance,
                            double reg_scale) {
  reg_scale = std::abs(reg_scale);
  auto px = points.select(-1, 0), py = points.select(-1, 1), pw = points.select(-1, 2),
       ph = points.select(-1, 3);
  auto dl = distance.select(-1, 0), dt = distance.select(-1, 1), dr = distance.select(-1, 2),
       db = distance.select(-1, 3);
  auto x1 = px - (0.5 * reg_scale + dl) * (pw / reg_scale);
  auto y1 = py - (0.5 * reg_scale + dt) * (ph / reg_scale);
  auto x2 = px + (0.5 * reg_scale + dr) * (pw / reg_scale);
  auto y2 = py + (0.5 * reg_scale + db) * (ph / reg_scale);
  return torch::stack({(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1}, -1);
}

// Integral over the corner distribution: softmax per edge then expectation under W(n).
torch::Tensor Integral(const torch::Tensor& corners, const torch::Tensor& project, int reg_max) {
  auto shape = corners.sizes().vec();
  auto x = torch::softmax(corners.reshape({-1, reg_max + 1}), 1);  // [N*4, reg_max+1]
  x = torch::matmul(x, project);                                   // [N*4]
  std::vector<std::int64_t> out(shape.begin(), shape.end() - 1);
  out.push_back(4);
  return x.reshape(out);
}

torch::Tensor InverseSigmoid(const torch::Tensor& x, double eps = 1e-5) {
  auto c = x.clamp(0.0, 1.0);
  return torch::log(c.clamp_min(eps) / (1 - c).clamp_min(eps));
}

// MS-deformable sampling: per level grid_sample the pre-split value at the box-scaled
// locations, weight by the (softmaxed) attention and sum. value[l]: [B,nhead,c,H*W].
torch::Tensor DeformCore(const std::vector<torch::Tensor>& value, const DfShapes& shapes,
                         const torch::Tensor& sampling_locations, const torch::Tensor& attn,
                         const std::vector<int>& num_points) {
  const auto B = value[0].size(0), nhead = value[0].size(1), c = value[0].size(2);
  const auto nq = sampling_locations.size(1);
  const int sum_points = std::accumulate(num_points.begin(), num_points.end(), 0);
  auto grids = (2 * sampling_locations - 1).permute({0, 2, 1, 3, 4}).flatten(0, 1);  // [B*nh,nq,sp,2]
  std::vector<std::int64_t> splits(num_points.begin(), num_points.end());
  auto grid_list = grids.split_with_sizes(splits, -2);
  std::vector<torch::Tensor> sampled;
  for (std::size_t lvl = 0; lvl < shapes.size(); ++lvl) {
    auto value_l = value[lvl].reshape({B * nhead, c, shapes[lvl].first, shapes[lvl].second});
    sampled.push_back(F::grid_sample(
        value_l, grid_list[lvl],
        F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false)));
  }
  auto aw = attn.permute({0, 2, 1, 3}).reshape({B * nhead, 1, nq, sum_points});
  auto out = (torch::cat(sampled, -1) * aw).sum(-1).reshape({B, nhead * c, nq});  // [B,d,nq]
  return out.permute({0, 2, 1});
}

}  // namespace

DfMLPImpl::DfMLPImpl(int in_dim, int hidden_dim, int out_dim, int num_layers, bool silu)
    : num_layers_(num_layers), silu_(silu) {
  layers = register_module("layers", nn::ModuleList());
  int prev = in_dim;
  for (int i = 0; i < num_layers; ++i) {
    const int out = (i == num_layers - 1) ? out_dim : hidden_dim;
    layers->push_back(nn::Linear(prev, out));
    prev = out;
  }
}

torch::Tensor DfMLPImpl::forward(torch::Tensor x) {
  for (int i = 0; i < num_layers_; ++i) {
    auto y = layers[static_cast<std::size_t>(i)]->as<nn::LinearImpl>()->forward(x);
    x = (i < num_layers_ - 1) ? (silu_ ? torch::silu(y) : torch::relu(y)) : y;
  }
  return x;
}

DfEncOutputImpl::DfEncOutputImpl(int d) {
  proj = register_module("proj", nn::Linear(d, d));
  norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({d})));
}
torch::Tensor DfEncOutputImpl::forward(torch::Tensor x) { return norm->forward(proj->forward(x)); }

DfRMSNormImpl::DfRMSNormImpl(int dim) {
  scale = register_parameter("scale", torch::ones({dim}));
}
torch::Tensor DfRMSNormImpl::forward(const torch::Tensor& x) {
  auto n = x * torch::rsqrt(x.pow(2).mean(-1, /*keepdim=*/true) + eps_);
  return n * scale;
}

DfSwiGLUImpl::DfSwiGLUImpl(int in_dim, int hidden_dim, int out_dim) {
  w12 = register_module("w12", nn::Linear(in_dim, 2 * hidden_dim));
  w3 = register_module("w3", nn::Linear(hidden_dim, out_dim));
}
torch::Tensor DfSwiGLUImpl::forward(const torch::Tensor& x) {
  auto c = w12->forward(x).chunk(2, -1);
  return w3->forward(torch::silu(c[0]) * c[1]);
}

DfDecInputProjImpl::DfDecInputProjImpl(int in_ch, int out_ch) : identity_(in_ch == out_ch) {
  if (!identity_) {
    conv = register_module("conv", nn::Conv2d(nn::Conv2dOptions(in_ch, out_ch, 1).bias(false)));
    norm = register_module("norm", nn::BatchNorm2d(out_ch));
  }
}
torch::Tensor DfDecInputProjImpl::forward(torch::Tensor x) {
  return identity_ ? x : norm->forward(conv->forward(x));
}

DfMSDeformImpl::DfMSDeformImpl(int d, int nhead, int num_levels, std::vector<int> num_points)
    : nhead_(nhead), num_levels_(num_levels), num_points_(std::move(num_points)) {
  const int sum_points = std::accumulate(num_points_.begin(), num_points_.end(), 0);
  total_points_ = nhead * sum_points;
  std::vector<float> nps;
  for (int n : num_points_) {
    for (int i = 0; i < n; ++i) nps.push_back(1.0F / static_cast<float>(n));
  }
  num_points_scale_ = register_buffer("num_points_scale", torch::tensor(nps));
  sampling_offsets = register_module("sampling_offsets", nn::Linear(d, total_points_ * 2));
  attention_weights = register_module("attention_weights", nn::Linear(d, total_points_));
}

torch::Tensor DfMSDeformImpl::forward(const torch::Tensor& query, const torch::Tensor& ref,
                                      const std::vector<torch::Tensor>& value,
                                      const DfShapes& shapes) {
  const auto B = query.size(0), nq = query.size(1);
  const int sum_points = total_points_ / nhead_;
  auto so = sampling_offsets->forward(query).reshape({B, nq, nhead_, sum_points, 2});
  auto aw = torch::softmax(attention_weights->forward(query).reshape({B, nq, nhead_, sum_points}), -1);
  auto nps = num_points_scale_.unsqueeze(-1);            // [sum_points, 1]
  auto ref_xy = ref.slice(-1, 0, 2).unsqueeze(2);        // [B, nq, 1, 1, 2]
  auto ref_wh = ref.slice(-1, 2, 4).unsqueeze(2);        // [B, nq, 1, 1, 2]
  auto sampling_locations = ref_xy + so * nps * ref_wh * offset_scale_;
  return DeformCore(value, shapes, sampling_locations, aw, num_points_);
}

DfGateImpl::DfGateImpl(int d, bool rms) {
  gate = register_module("gate", nn::Linear(2 * d, 2 * d));
  if (rms) {
    rmsnorm = register_module("norm", DfRMSNorm(d));
  } else {
    norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({d})));
  }
}
torch::Tensor DfGateImpl::forward(const torch::Tensor& x1, const torch::Tensor& x2) {
  auto g = torch::sigmoid(gate->forward(torch::cat({x1, x2}, -1)));
  auto c = g.chunk(2, -1);
  auto fused = c[0] * x1 + c[1] * x2;
  return rmsnorm ? rmsnorm->forward(fused) : norm->forward(fused);
}

DfDecLayerImpl::DfDecLayerImpl(int d, int nhead, int ffn, int num_levels,
                               std::vector<int> num_points, bool silu, bool deimv2)
    : silu_(silu), deimv2_(deimv2) {
  self_attn = register_module("self_attn",
                              nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, nhead).dropout(0.0)));
  cross_attn = register_module("cross_attn", DfMSDeform(d, nhead, num_levels, std::move(num_points)));
  gateway = register_module("gateway", DfGate(d, /*rms=*/deimv2));
  if (deimv2_) {
    // DEIMv2: RMSNorm + a SwiGLU FFN (hidden = ffn/2).
    rms1 = register_module("norm1", DfRMSNorm(d));
    swish_ffn = register_module("swish_ffn", DfSwiGLU(d, ffn / 2, d));
    rms3 = register_module("norm3", DfRMSNorm(d));
  } else {
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    linear1 = register_module("linear1", nn::Linear(d, ffn));
    linear2 = register_module("linear2", nn::Linear(ffn, d));
    norm3 = register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({d})));
  }
}

torch::Tensor DfDecLayerImpl::forward(torch::Tensor target, const torch::Tensor& ref,
                                      const std::vector<torch::Tensor>& value, const DfShapes& shapes,
                                      const torch::Tensor& query_pos) {
  auto q = (target + query_pos).transpose(0, 1);  // [L, N, d]
  auto v = target.transpose(0, 1);
  auto sa = std::get<0>(self_attn->forward(q, q, v)).transpose(0, 1);
  target = deimv2_ ? rms1->forward(target + sa) : norm1->forward(target + sa);
  auto ca = cross_attn->forward(target + query_pos, ref, value, shapes);
  target = gateway->forward(target, ca);
  torch::Tensor ffn;
  if (deimv2_) {
    ffn = swish_ffn->forward(target);
  } else {
    auto h = linear1->forward(target);
    ffn = linear2->forward(silu_ ? torch::silu(h) : torch::relu(h));
  }
  auto out = (target + ffn).clamp(-65504.0, 65504.0);
  return deimv2_ ? rms3->forward(out) : norm3->forward(out);
}

DfLQEImpl::DfLQEImpl(int k, int hidden_dim, int num_layers, int reg_max, bool silu)
    : k_(k), reg_max_(reg_max) {
  reg_conf = register_module("reg_conf", DfMLP(4 * (k + 1), hidden_dim, 1, num_layers, silu));
}
torch::Tensor DfLQEImpl::forward(const torch::Tensor& scores, const torch::Tensor& pred_corners) {
  const auto B = pred_corners.size(0), L = pred_corners.size(1);
  auto prob = torch::softmax(pred_corners.reshape({B, L, 4, reg_max_ + 1}), -1);
  auto topk = std::get<0>(prob.topk(k_, -1));
  auto stat = torch::cat({topk, topk.mean(-1, /*keepdim=*/true)}, -1);
  return scores + reg_conf->forward(stat.reshape({B, L, -1}));
}

DfDecoderStackImpl::DfDecoderStackImpl(int num_layers, int d, int nhead, int ffn, int num_levels,
                                       std::vector<int> num_points, int reg_max, bool silu,
                                       bool deimv2) {
  layers = register_module("layers", nn::ModuleList());
  lqe_layers = register_module("lqe_layers", nn::ModuleList());
  for (int i = 0; i < num_layers; ++i) {
    layers->push_back(DfDecLayer(d, nhead, ffn, num_levels, num_points, silu, deimv2));
    lqe_layers->push_back(DfLQE(4, 64, 2, reg_max, silu));
  }
}

DFINETransformerImpl::DFINETransformerImpl(DfTransformerConfig cfg) : cfg_(std::move(cfg)) {
  const int d = cfg_.hidden_dim, nc = cfg_.num_classes;
  project_ = WeightingFunction(cfg_.reg_max, cfg_.up, cfg_.reg_scale);

  input_proj = register_module("input_proj", nn::ModuleList());
  for (int fc : cfg_.feat_channels) {
    input_proj->push_back(DfDecInputProj(fc, d));
  }
  const bool silu = cfg_.silu;
  // DEIMv2 drops the enc_output projection (memory is scored directly) and uses a
  // 3-layer query_pos_head; D-FINE keeps enc_output and a 2-layer query_pos_head.
  if (!cfg_.deimv2) {
    enc_output = register_module("enc_output", DfEncOutput(d));
  }
  enc_score_head = register_module("enc_score_head", nn::Linear(d, nc));
  enc_bbox_head = register_module("enc_bbox_head", DfMLP(d, d, 4, 3, silu));
  query_pos_head = register_module(
      "query_pos_head", cfg_.deimv2 ? DfMLP(4, d, d, 3, silu) : DfMLP(4, 2 * d, d, 2, silu));
  pre_bbox_head = register_module("pre_bbox_head", DfMLP(d, d, 4, 3, silu));
  dec_score_head = register_module("dec_score_head", nn::ModuleList());
  dec_bbox_head = register_module("dec_bbox_head", nn::ModuleList());
  for (int i = 0; i < cfg_.num_layers; ++i) {
    dec_score_head->push_back(nn::Linear(d, nc));
    dec_bbox_head->push_back(DfMLP(d, d, 4 * (cfg_.reg_max + 1), 3, silu));
  }
  decoder = register_module(
      "decoder", DfDecoderStack(cfg_.num_layers, d, cfg_.nhead, cfg_.dim_feedforward,
                                cfg_.num_levels, cfg_.num_points, cfg_.reg_max, silu, cfg_.deimv2));
}

std::pair<torch::Tensor, torch::Tensor> DFINETransformerImpl::forward(
    const std::vector<torch::Tensor>& feats) {
  // Encoder input: flatten each (already-hidden_dim) level into one memory sequence.
  DfShapes shapes;
  std::vector<torch::Tensor> flat;
  for (std::size_t i = 0; i < feats.size(); ++i) {
    auto f = input_proj[i]->as<DfDecInputProjImpl>()->forward(feats[i]);  // neck -> decoder width
    shapes.emplace_back(f.size(2), f.size(3));
    flat.push_back(f.flatten(2).permute({0, 2, 1}));  // [B, hw, d]
  }
  auto memory = torch::cat(flat, 1).contiguous();  // [B, L, d]
  const auto B = memory.size(0);
  const int d = cfg_.hidden_dim;
  const auto project = project_.to(memory.options());

  // Two-stage query selection. The validity mask zeros invalid grid-anchor tokens for
  // the encoder scoring only; the deformable value below uses the unmasked memory.
  // D-FINE projects the masked memory through enc_output; DEIMv2 scores it directly.
  auto [anchors, valid_mask] = GenerateAnchors(shapes, cfg_.eps, memory.options());
  auto masked = valid_mask.to(memory.dtype()) * memory;
  auto output_memory = cfg_.deimv2 ? masked : enc_output->forward(masked);
  auto enc_logits = enc_score_head->forward(output_memory);  // [B, L, nc]
  auto topk = std::get<1>(std::get<0>(enc_logits.max(-1)).topk(cfg_.num_queries, -1));  // [B, nq]
  topk_idx_ = topk;
  auto gather_idx = topk.unsqueeze(-1);
  auto topk_memory = output_memory.gather(1, gather_idx.expand({B, cfg_.num_queries, d}));
  auto topk_anchors = anchors.expand({B, -1, -1}).gather(1, gather_idx.expand({B, cfg_.num_queries, 4}));
  auto ref_unact = (enc_bbox_head->forward(topk_memory) + topk_anchors);  // [B, nq, 4]
  auto content = topk_memory;

  // Pre-split value for the deformable attention (shared across layers).
  auto value = memory.reshape({B, memory.size(1), cfg_.nhead, d / cfg_.nhead}).permute({0, 2, 3, 1});
  std::vector<torch::Tensor> value_split;
  {
    std::vector<std::int64_t> sizes;
    for (const auto& hw : shapes) sizes.push_back(hw.first * hw.second);
    for (auto& v : value.split_with_sizes(sizes, -1)) value_split.push_back(v.contiguous());
  }

  // FDR decoder loop (eval: run all layers, emit the last).
  auto ref_points_detach = torch::sigmoid(ref_unact);
  torch::Tensor output = content, output_detach, pred_corners_undetach, ref_points_initial;
  torch::Tensor out_logits, out_boxes;
  for (int i = 0; i < cfg_.num_layers; ++i) {
    auto ref_input = ref_points_detach.unsqueeze(2);  // [B, nq, 1, 4]
    auto query_pos = query_pos_head->forward(ref_points_detach).clamp(-10.0, 10.0);
    output = decoder->layers[static_cast<std::size_t>(i)]->as<DfDecLayerImpl>()->forward(
        output, ref_input, value_split, shapes, query_pos);
    if (i == 0) {
      auto pre = torch::sigmoid(pre_bbox_head->forward(output) + InverseSigmoid(ref_points_detach));
      ref_points_initial = pre;
    }
    auto bbox_in = output_detach.defined() ? output + output_detach : output;
    auto pred_corners = dec_bbox_head[static_cast<std::size_t>(i)]->as<DfMLPImpl>()->forward(bbox_in);
    if (pred_corners_undetach.defined()) pred_corners = pred_corners + pred_corners_undetach;
    auto inter_ref = Distance2Bbox(ref_points_initial, Integral(pred_corners, project, cfg_.reg_max),
                                   cfg_.reg_scale);
    if (i == cfg_.num_layers - 1) {
      auto scores = dec_score_head[static_cast<std::size_t>(i)]->as<nn::LinearImpl>()->forward(output);
      scores = decoder->lqe_layers[static_cast<std::size_t>(i)]->as<DfLQEImpl>()->forward(scores, pred_corners);
      out_logits = scores;
      out_boxes = inter_ref;
      break;
    }
    pred_corners_undetach = pred_corners;
    ref_points_detach = inter_ref;
    output_detach = output;
  }
  return {out_logits, out_boxes};
}

}  // namespace detr::models
