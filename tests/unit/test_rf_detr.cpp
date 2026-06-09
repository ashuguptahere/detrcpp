// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RF-DETR smoke: a ViT backbone + multi-scale projection + query selection +
// deformable decoder forwards to the right shapes and trains with the focal path.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/models/registry.hpp"
#include "detr/train/target.hpp"
#include "detr/train/trainer.hpp"

namespace detr::models {
namespace {

YAML::Node Tiny() {
  YAML::Node c;
  c["vit_embed"] = 32;
  c["vit_depth"] = 2;
  c["vit_heads"] = 4;
  c["patch"] = 16;
  c["hidden_dim"] = 32;
  c["nheads"] = 4;
  c["dec_layers"] = 2;
  c["dim_feedforward"] = 64;
  c["num_queries"] = 8;
  c["num_classes"] = 5;
  c["num_levels"] = 3;
  c["imgsz"] = 64;
  return c;
}

class RfDetrTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(RfDetrTest, RegistersAndForwards) {
  EXPECT_TRUE(Registry::Instance().Contains("rf-detr"));
  auto built = Registry::Instance().Build("rf-detr", Tiny());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  EXPECT_TRUE(model->Meta().focal);
  model->eval();
  torch::NoGradGuard ng;
  // imgsz 64, patch 16 -> ViT 4x4; projector -> 4x4, 2x2, 1x1 (3 levels) = 21 tokens.
  auto out = model->Forward(torch::randn({1, 3, 64, 64}));
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 8, 5}));
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 8, 4}));
  EXPECT_GE(out.boxes.min().item<float>(), 0.0F);
  EXPECT_LE(out.boxes.max().item<float>(), 1.0F);
}

TEST_F(RfDetrTest, FocalTrainingStep) {
  auto model = *Registry::Instance().Build("rf-detr", Tiny());
  train::TrainConfig tc;
  tc.lr = 1e-4;
  train::Trainer trainer(model, tc);
  auto images = torch::randn({1, 3, 64, 64});
  train::Target t;
  t.labels = torch::tensor({1L, 3L});
  t.boxes = torch::tensor({{0.3F, 0.3F, 0.2F, 0.2F}, {0.7F, 0.6F, 0.25F, 0.3F}});
  const float loss = trainer.TrainStep(images, {t});
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_GE(loss, 0.0F);
}

}  // namespace
}  // namespace detr::models
