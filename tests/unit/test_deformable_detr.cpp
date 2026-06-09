// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Deformable-DETR forward smoke: multi-scale deformable encoder/decoder produces
// the right output shapes. (Faithful training/eval use a sigmoid focal head +
// sigmoid postprocess — tracked separately.)

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
  c["dec_layers"] = 1;
  c["dim_feedforward"] = 64;
  c["num_queries"] = 8;
  c["num_classes"] = 5;
  c["num_levels"] = 4;
  c["num_points"] = 4;
  c["imgsz"] = 64;
  return c;
}

class DeformableDetrTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(DeformableDetrTest, RegistersAndForwards) {
  EXPECT_TRUE(Registry::Instance().Contains("deformable-detr"));
  auto built = Registry::Instance().Build("deformable-detr", Tiny());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  model->eval();
  torch::NoGradGuard ng;
  // imgsz 64 -> C3 8x8, C4 4x4, C5 2x2, extra 1x1 (4 deformable levels).
  auto out = model->Forward(torch::randn({1, 3, 64, 64}));
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 8, 5}));  // num_classes (no +1)
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 8, 4}));
  EXPECT_GE(out.boxes.min().item<float>(), 0.0F);
  EXPECT_LE(out.boxes.max().item<float>(), 1.0F);
}

TEST_F(DeformableDetrTest, FocalTrainingStep) {
  auto model = *Registry::Instance().Build("deformable-detr", Tiny());
  EXPECT_TRUE(model->Meta().focal);  // sigmoid-focal classification
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
