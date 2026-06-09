// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RT-DETR smoke: the hybrid encoder (AIFI + CCFM) + query selection + deformable
// decoder produces the right shapes, and it trains with the focal path.

#include <gtest/gtest.h>
#include <torch/torch.h>

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
  c["num_levels"] = 3;
  c["num_points"] = 4;
  c["imgsz"] = 64;
  return c;
}

class RtDetrTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(RtDetrTest, RegistersAndForwards) {
  EXPECT_TRUE(Registry::Instance().Contains("rt-detr"));
  auto built = Registry::Instance().Build("rt-detr", Tiny());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  EXPECT_TRUE(model->Meta().focal);
  model->eval();
  torch::NoGradGuard ng;
  // imgsz 64 -> C3 8x8, C4 4x4, C5 2x2 (3 levels for the hybrid encoder).
  auto out = model->Forward(torch::randn({1, 3, 64, 64}));
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 8, 5}));
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 8, 4}));
  EXPECT_GE(out.boxes.min().item<float>(), 0.0F);
  EXPECT_LE(out.boxes.max().item<float>(), 1.0F);
}

TEST_F(RtDetrTest, SizeVariantsForward) {
  // s/n use a ResNet-18 (BasicBlock) backbone; l/x use Bottleneck. All must
  // forward to the same head shapes.
  for (const char* name :
       {"rt-detr-n", "rt-detr-s", "rt-detr-m", "rt-detr-x", "rt-detrv2-l", "rt-detrv3-s"}) {
    EXPECT_TRUE(Registry::Instance().Contains(name)) << name;
    YAML::Node c = Tiny();
    auto built = Registry::Instance().Build(name, c);
    ASSERT_TRUE(built.has_value()) << name << ": " << built.error().message;
    auto model = *built;
    model->eval();
    torch::NoGradGuard ng;
    auto out = model->Forward(torch::randn({1, 3, 64, 64}));
    EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 8, 5})) << name;
    EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 8, 4})) << name;
  }
}

TEST_F(RtDetrTest, FocalTrainingStep) {
  auto model = *Registry::Instance().Build("rt-detr", Tiny());
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
