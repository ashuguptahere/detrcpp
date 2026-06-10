// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DN-DETR: the DAB network plus denoising training. The denoising queries +
// group-isolation mask + reconstruction loss are train-only; inference is the
// plain DAB forward. Verified by an overfit-a-tiny-batch step.

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
  c["hidden_dim"] = 32;
  c["nheads"] = 4;
  c["enc_layers"] = 1;
  c["dec_layers"] = 2;
  c["dim_feedforward"] = 64;
  c["num_queries"] = 8;
  c["num_classes"] = 5;
  c["imgsz"] = 64;
  return c;
}

train::Target TinyTarget() {
  train::Target t;
  t.labels = torch::tensor({1L, 3L});
  t.boxes = torch::tensor({{0.3F, 0.3F, 0.2F, 0.2F}, {0.7F, 0.6F, 0.25F, 0.3F}});
  return t;
}

class DnDetrTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(DnDetrTest, RegistersAndSupportsDenoising) {
  EXPECT_TRUE(Registry::Instance().Contains("dn-detr"));
  auto built = Registry::Instance().Build("dn-detr", Tiny());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  EXPECT_EQ(model->Meta().name, "dn-detr");
  EXPECT_TRUE(model->Meta().focal);
  EXPECT_TRUE(model->SupportsDenoising());
}

// Inference is the plain DAB forward: only the matching queries, no DN columns,
// no aux.
TEST_F(DnDetrTest, InferenceForwardReturnsOnlyMatchingQueries) {
  auto model = *Registry::Instance().Build("dn-detr", Tiny());
  model->eval();
  torch::NoGradGuard ng;
  auto out = model->Forward(torch::randn({1, 3, 64, 64}));
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 8, 5}));
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 8, 4}));
  EXPECT_TRUE(out.aux_logits.empty());
  EXPECT_GE(out.boxes.min().item<float>(), 0.0F);
  EXPECT_LE(out.boxes.max().item<float>(), 1.0F);
}

TEST_F(DnDetrTest, DenoisingTrainingStep) {
  auto model = *Registry::Instance().Build("dn-detr", Tiny());
  train::TrainConfig tc;
  tc.lr = 1e-4;
  tc.seed = 0;
  train::Trainer trainer(model, tc);  // dn defaults: dn_number=5
  auto t = TinyTarget();
  const float loss = trainer.TrainStep(torch::randn({1, 3, 64, 64}), {t});
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_GT(loss, 0.0F);
}

TEST_F(DnDetrTest, OverfitsTinyBatch) {
  torch::manual_seed(0);  // deterministic init + batch + training (no flaky threshold)
  auto model = *Registry::Instance().Build("dn-detr", Tiny());
  train::TrainConfig tc;
  tc.lr = 2e-3;
  tc.seed = 0;
  tc.grad_clip = 0.1;
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
