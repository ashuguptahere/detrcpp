// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Deformable-DETR forward smoke: multi-scale deformable encoder/decoder produces
// the right output shapes. (Faithful training/eval use a sigmoid focal head +
// sigmoid postprocess — tracked separately.)

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/models/registry.hpp"

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

}  // namespace
}  // namespace detr::models
