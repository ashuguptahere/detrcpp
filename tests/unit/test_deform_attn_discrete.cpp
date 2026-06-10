// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RT-DETRv2 discrete-sampling variant of multi-scale deformable attention: rounds
// each sampling location to the nearest pixel and gathers (vs bilinear interp).
// Default-off, so dino/rf-detr/deformable + rt-detr/rt-detrv3 stay bilinear.

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

// Gating end-to-end: rt-detrv2 (discrete) differs from rt-detr (bilinear) at the
// SAME weights, while rt-detrv3 (also bilinear) is identical to rt-detr.
class RtDetrDiscreteTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(RtDetrDiscreteTest, V2DiscreteDiffersV1V3Identical) {
  auto images = torch::randn({1, 3, 64, 64});
  auto run = [&](const char* name) {
    torch::manual_seed(42);  // same init for each (same module tree)
    auto m = *Registry::Instance().Build(name, Tiny());
    m->eval();
    torch::NoGradGuard ng;
    return m->Forward(images).boxes;
  };
  auto v1 = run("rt-detr-l");
  auto v2 = run("rt-detrv2-l");
  auto v3 = run("rt-detrv3-l");
  EXPECT_TRUE(torch::isfinite(v2).all().item<bool>());
  EXPECT_TRUE(torch::allclose(v1, v3));                      // both bilinear -> identical
  EXPECT_GT((v2 - v1).abs().max().item<float>(), 1e-5F);     // v2 discrete -> differs
}

}  // namespace
}  // namespace detr::models
