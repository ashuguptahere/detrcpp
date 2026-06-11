// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DINO-CDN: DINO + contrastive denoising training. Per GT, a positive (small box
// noise -> reconstruct GT) and a negative (large box noise -> no-object) query
// run through the shared deformable decoder under a group-isolation mask. The
// denoising path is train-only; inference is plain DINO.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "detr/models/registry.hpp"
#include "detr/train/denoising.hpp"
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
  c["num_queries"] = 20;
  c["num_classes"] = 5;
  c["num_levels"] = 4;
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

class DinoCdnTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(DinoCdnTest, RegistersAndSupportsDenoising) {
  EXPECT_TRUE(Registry::Instance().Contains("dino-cdn"));
  auto built = Registry::Instance().Build("dino-cdn", Tiny());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  EXPECT_EQ(model->Meta().name, "dino-cdn");
  EXPECT_TRUE(model->Meta().focal);
  EXPECT_TRUE(model->SupportsDenoising());
}

TEST_F(DinoCdnTest, InferenceForwardReturnsOnlyMatchingQueries) {
  auto model = *Registry::Instance().Build("dino-cdn", Tiny());
  model->eval();
  torch::NoGradGuard ng;
  auto out = model->Forward(torch::randn({1, 3, 64, 64}));
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 20, 5}));
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 20, 4}));
  EXPECT_TRUE(out.aux_logits.empty());
  EXPECT_GE(out.boxes.min().item<float>(), 0.0F);
  EXPECT_LE(out.boxes.max().item<float>(), 1.0F);
}

// The contrastive layout: 2*T queries per group (positives matchable, negatives
// present-but-unmatched), one [L,L] mask, the known assignment matches only
// positives.
TEST_F(DinoCdnTest, ContrastiveDenoisingLayout) {
  train::DnConfig dn;
  dn.dn_number = 5;
  dn.contrastive = true;
  const train::TargetBatch batch{TinyTarget()};  // T=2
  auto [in, layout] = train::MakeDenoising(batch, dn, /*num_classes=*/5, /*num_queries=*/20);
  ASSERT_TRUE(in.active);
  EXPECT_EQ(in.num_dn, 5 * 2 * 2);  // dn_number * 2 (pos+neg) * t_pad
  EXPECT_EQ(in.dn_ref.size(0), 20);
  EXPECT_EQ(layout.pad_mask.sum().item<std::int64_t>(), 10);  // positives only (G*T)
  auto matches = train::BuildDnMatches(layout);
  ASSERT_EQ(matches.size(), 1U);
  EXPECT_EQ(matches[0].first.numel(), 10);   // src = positive query indices
  EXPECT_EQ(matches[0].second.numel(), 10);  // tgt = GT indices
}

TEST_F(DinoCdnTest, DenoisingTrainingStep) {
  auto model = *Registry::Instance().Build("dino-cdn", Tiny());
  train::TrainConfig tc;
  tc.lr = 1e-4;
  tc.seed = 0;
  tc.dn.contrastive = true;
  tc.dn.box_noise_scale = 1.0;
  tc.dn.label_noise_ratio = 0.5;
  train::Trainer trainer(model, tc);
  const float loss = trainer.TrainStep(torch::randn({1, 3, 64, 64}), {TinyTarget()});
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_GT(loss, 0.0F);
}

TEST_F(DinoCdnTest, OverfitsTinyBatch) {
  torch::manual_seed(0);  // deterministic init + batch + training (no flaky threshold)
  auto model = *Registry::Instance().Build("dino-cdn", Tiny());
  train::TrainConfig tc;
  tc.lr = 2e-3;
  tc.seed = 0;
  tc.grad_clip = 0.1;
  tc.dn.contrastive = true;
  train::Trainer trainer(model, tc);

  auto images = torch::randn({1, 3, 64, 64});
  const train::TargetBatch batch{TinyTarget()};
  const float first = trainer.TrainStep(images, batch);
  float last = first;
  for (int i = 0; i < 60; ++i) {
    last = trainer.TrainStep(images, batch);
  }
  EXPECT_TRUE(std::isfinite(last));
  EXPECT_LT(last, first * 0.7F) << "first=" << first << " last=" << last;
}

}  // namespace
}  // namespace detr::models
