// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/pos_embed.hpp"

#include <torch/torch.h>

#include <numbers>

namespace detr::models {

torch::Tensor SinePos(std::int64_t b, std::int64_t d, std::int64_t h, std::int64_t w,
                      const torch::TensorOptions& opts, double temperature) {
  const std::int64_t half = d / 2;
  const double scale = 2.0 * std::numbers::pi;
  auto ys = torch::arange(1, h + 1, opts) / (static_cast<double>(h) + 1e-6) * scale;
  auto xs = torch::arange(1, w + 1, opts) / (static_cast<double>(w) + 1e-6) * scale;
  auto dim_t = torch::arange(0, half, opts);
  dim_t = torch::pow(temperature, 2 * torch::floor(dim_t / 2) / static_cast<double>(half));

  auto px = xs.unsqueeze(1) / dim_t.unsqueeze(0);
  auto py = ys.unsqueeze(1) / dim_t.unsqueeze(0);
  auto interleave = [half](torch::Tensor p) {
    return torch::stack({p.slice(1, 0, half, 2).sin(), p.slice(1, 1, half, 2).cos()}, 2).flatten(1);
  };
  px = interleave(px);
  py = interleave(py);
  auto pyf = py.unsqueeze(1).expand({h, w, half});
  auto pxf = px.unsqueeze(0).expand({h, w, half});
  auto pos = torch::cat({pyf, pxf}, 2);
  return pos.permute({2, 0, 1}).unsqueeze(0).expand({b, d, h, w}).contiguous();
}

}  // namespace detr::models
