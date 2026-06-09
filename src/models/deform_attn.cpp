// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/deform_attn.hpp"

#include <cstdint>
#include <vector>

namespace detr::models {

namespace {
namespace nn = torch::nn;
namespace F = torch::nn::functional;
}  // namespace

torch::Tensor MSDeformAttnCore(const torch::Tensor& value, const SpatialShapes& shapes,
                               const torch::Tensor& sampling_locations,
                               const torch::Tensor& attention_weights) {
  const auto n = value.size(0);
  const auto heads = value.size(2);
  const auto dim = value.size(3);
  const auto lq = sampling_locations.size(1);
  const auto levels = sampling_locations.size(3);
  const auto points = sampling_locations.size(4);

  std::vector<std::int64_t> split_sizes;
  split_sizes.reserve(shapes.size());
  for (const auto& [h, w] : shapes) {
    split_sizes.push_back(h * w);
  }
  const auto value_list = value.split_with_sizes(split_sizes, 1);
  const auto sampling_grids = 2 * sampling_locations - 1;  // [0,1] -> [-1,1]

  const auto sample_opts = F::GridSampleFuncOptions()
                               .mode(torch::kBilinear)
                               .padding_mode(torch::kZeros)
                               .align_corners(false);
  std::vector<torch::Tensor> sampling_value_list;
  sampling_value_list.reserve(shapes.size());
  for (std::size_t lid = 0; lid < shapes.size(); ++lid) {
    const auto h = shapes[lid].first;
    const auto w = shapes[lid].second;
    // [N, H*W, M, D] -> [N, M*D, H*W] -> [N*M, D, H, W]
    auto value_l = value_list[lid].flatten(2).transpose(1, 2).reshape({n * heads, dim, h, w});
    // [N, Lq, M, P, 2] for this level -> [N, M, Lq, P, 2] -> [N*M, Lq, P, 2]
    auto grid_l =
        sampling_grids.select(3, static_cast<std::int64_t>(lid)).transpose(1, 2).flatten(0, 1);
    // -> [N*M, D, Lq, P]
    sampling_value_list.push_back(F::grid_sample(value_l, grid_l, sample_opts));
  }

  // [N, Lq, M, L, P] -> [N, M, Lq, L, P] -> [N*M, 1, Lq, L*P]
  auto attn = attention_weights.transpose(1, 2).reshape({n * heads, 1, lq, levels * points});
  // stack -> [N*M, D, Lq, L, P] -> [N*M, D, Lq, L*P]
  auto stacked = torch::stack(sampling_value_list, -2).flatten(-2);
  auto output = (stacked * attn).sum(-1).view({n, heads * dim, lq});
  return output.transpose(1, 2).contiguous();  // [N, Lq, M*D]
}

MSDeformAttnImpl::MSDeformAttnImpl(int d_model, int n_levels, int n_heads, int n_points)
    : d_model_(d_model), n_levels_(n_levels), n_heads_(n_heads), n_points_(n_points) {
  sampling_offsets =
      register_module("sampling_offsets", nn::Linear(d_model, n_heads * n_levels * n_points * 2));
  attention_weights =
      register_module("attention_weights", nn::Linear(d_model, n_heads * n_levels * n_points));
  value_proj = register_module("value_proj", nn::Linear(d_model, d_model));
  output_proj = register_module("output_proj", nn::Linear(d_model, d_model));
}

torch::Tensor MSDeformAttnImpl::forward(const torch::Tensor& query,
                                        const torch::Tensor& reference_points,
                                        const torch::Tensor& input_flatten,
                                        const SpatialShapes& shapes) {
  const auto n = query.size(0);
  const auto lq = query.size(1);
  const auto lv = input_flatten.size(1);
  const auto head_dim = d_model_ / n_heads_;

  auto value = value_proj->forward(input_flatten).view({n, lv, n_heads_, head_dim});

  auto offsets = sampling_offsets->forward(query).view({n, lq, n_heads_, n_levels_, n_points_, 2});
  auto attn = attention_weights->forward(query).view({n, lq, n_heads_, n_levels_ * n_points_});
  attn = attn.softmax(-1).view({n, lq, n_heads_, n_levels_, n_points_});

  // Sampling locations from the reference points. 2D refs (cx,cy): offset is
  // normalized by each level's (W,H). 4D refs (cx,cy,w,h): offset is scaled by
  // the box (w,h) — RT-DETR / iterative-refinement decoders.
  torch::Tensor sampling_locations;
  if (reference_points.size(-1) == 2) {
    std::vector<std::int64_t> wh;
    wh.reserve(shapes.size() * 2);
    for (const auto& [h, w] : shapes) {
      wh.push_back(w);
      wh.push_back(h);
    }
    auto offset_normalizer = torch::tensor(wh, query.options().dtype(torch::kLong))
                                 .view({n_levels_, 2})
                                 .to(query.dtype());
    sampling_locations = reference_points.view({n, lq, 1, n_levels_, 1, 2}) +
                         offsets / offset_normalizer.view({1, 1, 1, n_levels_, 1, 2});
  } else {  // 4D
    auto ref = reference_points.view({n, lq, 1, n_levels_, 1, 4});
    sampling_locations =
        ref.slice(-1, 0, 2) + offsets / static_cast<double>(n_points_) * ref.slice(-1, 2, 4) * 0.5;
  }

  auto out = MSDeformAttnCore(value, shapes, sampling_locations, attn);
  return output_proj->forward(out);
}

}  // namespace detr::models
