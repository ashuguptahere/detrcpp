// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RT-DETRv2's optional discrete-sampling deformable attention: rounds each sampling
// location to the nearest pixel and gathers (vs bilinear interp). It is the `_dsp`
// deployment mode — opt-in via `discrete_sample: true` — so every registered model
// (including the headline rt-detrv2, whose official config is `cross_attn_method:
// default`) stays bilinear/grid by default.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/models/deform_attn.hpp"
#include "detr/models/registry.hpp"

namespace detr::models {
namespace {

YAML::Node Tiny() {
  YAML::Node c;
  c["backbone"] = "r18";
  c["hidden_dim"] = 32;
  c["nheads"] = 4;
  c["enc_layers"] = 1;
  c["dec_layers"] = 2;
  c["dim_feedforward"] = 64;
  c["num_queries"] = 8;
  c["num_classes"] = 5;
  c["num_levels"] = 3;
  c["num_points"] = 4;
  c["imgsz"] = 64;
  return c;
}

// The sampling core: discrete produces the right shape, is finite, and differs
// from bilinear (it rounds — no parity by design).
TEST(DeformAttnDiscrete, CoreDiscreteVsBilinear) {
  torch::manual_seed(0);
  const SpatialShapes shapes{{4, 4}};
  auto value = torch::randn({1, 16, 1, 2});                  // [N, Sum(HW), heads, dim]
  auto loc = torch::rand({1, 3, 1, 1, 2, 2});                // [N, Lq, heads, levels, points, 2]
  auto attn = torch::rand({1, 3, 1, 1, 2}).softmax(-1);      // [N, Lq, heads, levels, points]

  auto bilinear = MSDeformAttnCore(value, shapes, loc, attn, /*discrete=*/false);
  auto discrete = MSDeformAttnCore(value, shapes, loc, attn, /*discrete=*/true);
  EXPECT_EQ(bilinear.sizes(), (std::vector<std::int64_t>{1, 3, 2}));
  EXPECT_EQ(discrete.sizes(), bilinear.sizes());
  EXPECT_TRUE(torch::isfinite(discrete).all().item<bool>());
  EXPECT_GT((discrete - bilinear).abs().max().item<float>(), 1e-4F);  // rounds != interps

  // The default argument is the bilinear path (byte-identical to explicit false).
  auto deflt = MSDeformAttnCore(value, shapes, loc, attn);
  EXPECT_TRUE(torch::allclose(deflt, bilinear));
}

// Gating end-to-end: every RT-DETR variant defaults to grid sampling (the official
// headline configs), so v1/v2/v3 share one inference graph at the same init; opting
// into discrete sampling (the `_dsp` mode) changes the output.
class RtDetrDiscreteTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(RtDetrDiscreteTest, GridDefaultDiscreteOptIn) {
  auto images = torch::randn({1, 3, 64, 64});
  auto run = [&](const char* name, bool discrete) {
    torch::manual_seed(42);  // same init for each (same module tree)
    auto cfg = Tiny();
    if (discrete) {
      cfg["discrete_sample"] = true;
    }
    auto m = *Registry::Instance().Build(name, cfg);
    m->eval();
    torch::NoGradGuard ng;
    return m->Forward(images).boxes;
  };
  auto v1 = run("rt-detr-l", false);
  auto v2 = run("rt-detrv2-l", false);
  auto v3 = run("rt-detrv3-l", false);
  EXPECT_TRUE(torch::allclose(v1, v2));  // all grid by default -> identical graph
  EXPECT_TRUE(torch::allclose(v1, v3));
  // Discrete sampling is opt-in (v2's `_dsp` deployment mode) and changes the output.
  auto v2_dsp = run("rt-detrv2-l", true);
  EXPECT_TRUE(torch::isfinite(v2_dsp).all().item<bool>());
  EXPECT_GT((v2_dsp - v1).abs().max().item<float>(), 1e-5F);
}

}  // namespace
}  // namespace detr::models
