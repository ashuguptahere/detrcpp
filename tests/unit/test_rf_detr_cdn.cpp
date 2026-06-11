// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RF-DETR-CDN: RF-DETR (ViT backbone + shared deformable head) + contrastive
// denoising training. Reuses the same CDN path as DINO-CDN; train-only.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "detr/models/registry.hpp"
#include "detr/train/target.hpp"
#include "detr/train/trainer.hpp"

namespace detr::models {
namespace {

// Kept deliberately small (patch 16 -> a 4x4 ViT grid, 1 block) so the overfit
// loop stays fast — the ViT backbone is the cost.
YAML::Node Tiny() {
  YAML::Node c;
  c["vit_embed"] = 32;
  c["vit_depth"] = 1;
  c["vit_heads"] = 4;
  c["patch"] = 16;
  c["hidden_dim"] = 32;
  c["nheads"] = 4;
  c["dec_layers"] = 2;
  c["dim_feedforward"] = 64;
  c["num_queries"] = 10;
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

class RfDetrCdnTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(RfDetrCdnTest, RegistersAndSupportsDenoising) {
  EXPECT_TRUE(Registry::Instance().Contains("rf-detr-cdn"));
  auto built = Registry::Instance().Build("rf-detr-cdn", Tiny());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  EXPECT_EQ(model->Meta().name, "rf-detr-cdn");
  EXPECT_TRUE(model->Meta().focal);
  EXPECT_TRUE(model->SupportsDenoising());
}

TEST_F(RfDetrCdnTest, InferenceForwardReturnsOnlyMatchingQueries) {
  auto model = *Registry::Instance().Build("rf-detr-cdn", Tiny());
  model->eval();
  torch::NoGradGuard ng;
  auto out = model->Forward(torch::randn({1, 3, 64, 64}));
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 10, 5}));
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 10, 4}));
  EXPECT_TRUE(out.aux_logits.empty());
}

TEST_F(RfDetrCdnTest, OverfitsTinyBatch) {
  torch::manual_seed(0);  // deterministic init + batch + training (no flaky threshold)
  auto model = *Registry::Instance().Build("rf-detr-cdn", Tiny());
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
  for (int i = 0; i < 25; ++i) {
    last = trainer.TrainStep(images, batch);
  }
  EXPECT_TRUE(std::isfinite(last));
  EXPECT_LT(last, first * 0.8F) << "first=" << first << " last=" << last;
}

}  // namespace
}  // namespace detr::models
