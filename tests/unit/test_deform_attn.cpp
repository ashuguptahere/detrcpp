// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Numerical parity for multi-scale deformable attention: compares MSDeformAttnCore
// against Deformable-DETR's ms_deform_attn_core_pytorch reference dumped by
// /tmp/gen_deform_ref.py (scripts/... generates it). Skips if the fixture is
// absent so the suite still runs without it.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <filesystem>

#include "detr/models/deform_attn.hpp"
#include "detr/weights/pth.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

TEST(DeformAttnTest, CoreMatchesTorchReference) {
  const std::filesystem::path ref = "/tmp/deform_ref.pth";
  if (!std::filesystem::exists(ref)) {
    GTEST_SKIP() << "reference fixture not present: " << ref;
  }
  auto sd = weights::LoadPth(ref);
  ASSERT_TRUE(sd.has_value()) << sd.error().message;

  auto get = [&](const char* name) {
    const auto* raw = sd->Find(name);
    EXPECT_NE(raw, nullptr) << "missing tensor: " << name;
    return weights::ToTensor(*raw);
  };
  auto value = get("value");
  auto sampling_locations = get("sampling_locations");
  auto attention_weights = get("attention_weights");
  auto expected = get("output");
  auto shapes_t = get("shapes");  // [n_levels, 2] int64 (H, W)

  SpatialShapes shapes;
  for (std::int64_t i = 0; i < shapes_t.size(0); ++i) {
    shapes.emplace_back(shapes_t[i][0].item<std::int64_t>(), shapes_t[i][1].item<std::int64_t>());
  }

  auto out = MSDeformAttnCore(value, shapes, sampling_locations, attention_weights);
  ASSERT_EQ(out.sizes(), expected.sizes());
  const double max_abs = (out - expected).abs().max().item<double>();
  EXPECT_LT(max_abs, 1e-5) << "max|Δ| = " << max_abs;
}

TEST(DeformAttnTest, ModuleForwardShapes) {
  const int d = 16, levels = 2, heads = 4, points = 2;
  MSDeformAttn attn(d, levels, heads, points);
  attn->eval();
  torch::NoGradGuard ng;

  SpatialShapes shapes{{4, 4}, {2, 2}};
  const std::int64_t lv = 16 + 4, lq = 5, n = 1;
  auto query = torch::randn({n, lq, d});
  auto ref = torch::rand({n, lq, levels, 2});
  auto input = torch::randn({n, lv, d});

  auto out = attn->forward(query, ref, input, shapes);
  EXPECT_EQ(out.sizes(), (std::vector<std::int64_t>{n, lq, d}));
}

}  // namespace
}  // namespace detr::models
