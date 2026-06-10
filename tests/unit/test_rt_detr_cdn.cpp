// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RT-DETR-CDN: RT-DETR (ResNet-VD + hybrid encoder + shared deformable head) with
// contrastive denoising training. Reuses the same CDN path as DINO/RF-DETR-CDN;
// train-only, inference is plain RT-DETR.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "detr/models/registry.hpp"
#include "detr/train/target.hpp"
#include "detr/train/trainer.hpp"

namespace detr::models {
namespace {

YAML::Node Tiny() {
  YAML::Node c;
  c["backbone"] = "r18";  // lighter than the default r50 for a fast test
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

class RtDetrCdnTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(RtDetrCdnTest, RegistersAndSupportsDenoising) {
  EXPECT_TRUE(Registry::Instance().Contains("rt-detr-cdn"));
  auto built = Registry::Instance().Build("rt-detr-cdn", Tiny());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  EXPECT_EQ(model->Meta().name, "rt-detr-cdn");
  EXPECT_TRUE(model->Meta().focal);
  EXPECT_TRUE(model->SupportsDenoising());
}

TEST_F(RtDetrCdnTest, InferenceForwardReturnsOnlyMatchingQueries) {
  auto model = *Registry::Instance().Build("rt-detr-cdn", Tiny());
  model->eval();
  torch::NoGradGuard ng;
  auto out = model->Forward(torch::randn({1, 3, 64, 64}));
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 8, 5}));
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 8, 4}));
  EXPECT_TRUE(out.aux_logits.empty());
}

TEST_F(RtDetrCdnTest, OverfitsTinyBatch) {
  torch::manual_seed(0);  // deterministic init + batch + training (no flaky threshold)
  auto model = *Registry::Instance().Build("rt-detr-cdn", Tiny());
  train::TrainConfig tc;
  tc.lr = 2e-3;
  tc.grad_clip = 0.1;
  tc.dn.contrastive = true;
  train::Trainer trainer(model, tc);

  auto images = torch::randn({1, 3, 64, 64});
  const train::TargetBatch batch{TinyTarget()};
  const float first = trainer.TrainStep(images, batch);
  float last = first;
  for (int i = 0; i < 40; ++i) {
    last = trainer.TrainStep(images, batch);
  }
  EXPECT_TRUE(std::isfinite(last));
  EXPECT_LT(last, first * 0.8F) << "first=" << first << " last=" << last;
}

}  // namespace
}  // namespace detr::models
