// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RT-DETRv3 hierarchical dense positive supervision: a one-to-many matcher (each
// GT supervises its top-k queries) adds an auxiliary loss for denser gradient.
// Enabled only for the rt-detrv3 matrix; train-only.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "detr/models/registry.hpp"
#include "detr/train/matcher.hpp"
#include "detr/train/target.hpp"
#include "detr/train/trainer.hpp"

namespace detr::models {
namespace {

YAML::Node Tiny() {
  YAML::Node c;
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

train::Target TinyTarget() {
  train::Target t;
  t.labels = torch::tensor({1L, 3L});
  t.boxes = torch::tensor({{0.3F, 0.3F, 0.2F, 0.2F}, {0.7F, 0.6F, 0.25F, 0.3F}});
  return t;
}

class RtDetrV3Test : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

// The one-to-many matcher gives each GT its top-k lowest-cost queries.
TEST_F(RtDetrV3Test, OneToManyMatchTopK) {
  Detections out;
  out.logits = torch::randn({1, 12, 5});  // Q=12, C=5
  out.boxes = torch::rand({1, 12, 4});
  train::MatchWeights w;
  w.focal = true;
  auto m = train::OneToManyMatch(out, {TinyTarget()}, /*k=*/3, w);  // T=2
  ASSERT_EQ(m.size(), 1U);
  EXPECT_EQ(m[0].first.numel(), 6);   // k*T = 3*2 src query indices
  EXPECT_EQ(m[0].second.numel(), 6);  // and 6 GT indices
  EXPECT_EQ((m[0].second == 0).sum().item<std::int64_t>(), 3);  // GT 0 supervised 3x
  EXPECT_EQ((m[0].second == 1).sum().item<std::int64_t>(), 3);  // GT 1 supervised 3x
}

TEST_F(RtDetrV3Test, DenseSupervisionEnabledForV3Only) {
  EXPECT_TRUE(Registry::Instance().Contains("rt-detrv3-l"));
  auto v3 = *Registry::Instance().Build("rt-detrv3-l", Tiny());
  EXPECT_EQ(v3->DenseSupervisionK(), 6);
  auto v1 = *Registry::Instance().Build("rt-detr-l", Tiny());
  EXPECT_EQ(v1->DenseSupervisionK(), 0);  // plain rt-detr: no dense supervision
  auto v2 = *Registry::Instance().Build("rt-detrv2-l", Tiny());
  EXPECT_EQ(v2->DenseSupervisionK(), 0);
}

TEST_F(RtDetrV3Test, DenseSupervisionTrainingStep) {
  auto model = *Registry::Instance().Build("rt-detrv3-l", Tiny());
  ASSERT_GT(model->DenseSupervisionK(), 0);
  train::TrainConfig tc;
  tc.lr = 1e-4;
  tc.seed = 0;
  train::Trainer trainer(model, tc);
  const float loss = trainer.TrainStep(torch::randn({1, 3, 64, 64}), {TinyTarget()});
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_GT(loss, 0.0F);
}

TEST_F(RtDetrV3Test, OverfitsTinyBatch) {
  torch::manual_seed(0);  // deterministic init + batch + training (no flaky threshold)
  auto model = *Registry::Instance().Build("rt-detrv3-l", Tiny());
  train::TrainConfig tc;
  tc.lr = 2e-3;
  tc.grad_clip = 0.1;
  train::Trainer trainer(model, tc);

  auto images = torch::randn({1, 3, 64, 64});
  const train::TargetBatch batch{TinyTarget()};
  const float first = trainer.TrainStep(images, batch);
  // Dense supervision adds the one-to-many loss (k=6) on every layer, so the loss
  // is larger and converges slower than the plain models — give it more steps.
  float last = first;
  for (int i = 0; i < 60; ++i) {
    last = trainer.TrainStep(images, batch);
  }
  EXPECT_TRUE(std::isfinite(last));
  EXPECT_LT(last, first * 0.8F) << "first=" << first << " last=" << last;
}

}  // namespace
}  // namespace detr::models
