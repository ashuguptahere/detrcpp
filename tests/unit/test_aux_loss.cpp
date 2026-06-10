// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Verifies DETR deep-supervision plumbing: in training mode the model emits one
// auxiliary (logits, boxes) per intermediate decoder layer, each shaped like the
// final prediction; in eval mode it emits none, so inference/postprocess and the
// validated eval path are unaffected.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstddef>

#include "detr/models/model.hpp"
#include "detr/models/registry.hpp"

namespace detr::models {
namespace {

YAML::Node MultiLayerConfig() {
  YAML::Node c;
  c["hidden_dim"] = 32;
  c["nheads"] = 4;
  c["enc_layers"] = 1;
  c["dec_layers"] = 3;  // -> 2 intermediate layers -> 2 aux outputs
  c["dim_feedforward"] = 64;
  c["num_queries"] = 5;
  c["num_classes"] = 4;
  c["backbone_width"] = 8;
  return c;
}

class AuxLossTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(AuxLossTest, TrainingEmitsPerLayerAuxEvalEmitsNone) {
  auto built = Registry::Instance().Build("detr", MultiLayerConfig());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  auto images = torch::randn({2, 3, 64, 64});

  // Eval: no auxiliary outputs (inference path is unchanged).
  model->eval();
  {
    torch::NoGradGuard ng;
    auto out = model->Forward(images);
    EXPECT_TRUE(out.aux_logits.empty());
    EXPECT_TRUE(out.aux_boxes.empty());
  }

  // Training: one aux per intermediate decoder layer (dec_layers - 1 == 2),
  // each shaped like the final prediction.
  model->train();
  auto out = model->Forward(images);
  ASSERT_EQ(out.aux_logits.size(), 2U);
  ASSERT_EQ(out.aux_boxes.size(), 2U);
  for (std::size_t i = 0; i < out.aux_logits.size(); ++i) {
    EXPECT_EQ(out.aux_logits[i].sizes(), out.logits.sizes());
    EXPECT_EQ(out.aux_boxes[i].sizes(), out.boxes.sizes());
  }
}

}  // namespace
}  // namespace detr::models
