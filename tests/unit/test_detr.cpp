// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Proves the DETR model: it registers, builds from a YAML config, runs a forward
// pass with the correct output shapes and box range, and round-trips its weights
// through the safetensors interchange into a freshly-built model.

#include "detr/models/detr.hpp"

#include <filesystem>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/models/registry.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

YAML::Node TinyConfig() {
  YAML::Node c;
  c["hidden_dim"] = 32;
  c["nheads"] = 4;
  c["enc_layers"] = 1;
  c["dec_layers"] = 1;
  c["dim_feedforward"] = 64;
  c["num_queries"] = 5;
  c["num_classes"] = 4;
  c["backbone_width"] = 8;
  return c;
}

// gtest_discover_tests runs each test in its own process, so every test must
// register the built-ins itself (Register is idempotent).
class DetrTest : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(DetrTest, RegistersInRegistry) {
  EXPECT_TRUE(Registry::Instance().Contains("detr"));
  bool found = false;
  for (const auto& m : Registry::Instance().List()) {
    if (m.name == "detr") {
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(DetrTest, ForwardShapesAndBoxRange) {
  auto built = Registry::Instance().Build("detr", TinyConfig());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  model->eval();
  torch::NoGradGuard ng;

  auto images = torch::randn({2, 3, 64, 64});
  auto out = model->Forward(images);

  // logits: [B, Q, num_classes + 1] = [2, 5, 5]; boxes: [2, 5, 4] in [0,1].
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{2, 5, 5}));
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{2, 5, 4}));
  EXPECT_GE(out.boxes.min().item<float>(), 0.0F);
  EXPECT_LE(out.boxes.max().item<float>(), 1.0F);
}

TEST_F(DetrTest, WeightsRoundTripThroughSafetensors) {
  auto a_built = Registry::Instance().Build("detr", TinyConfig());
  ASSERT_TRUE(a_built.has_value()) << a_built.error().message;
  auto a = *a_built;
  weights::StateDict sd = weights::StateDictFromModule(*a);
  EXPECT_GT(sd.Size(), 10U);  // backbone + transformer + heads have many tensors

  const auto path =
      std::filesystem::temp_directory_path() / "detr_model_roundtrip.safetensors";
  ASSERT_TRUE(weights::SaveSafetensors(path, sd).has_value());
  auto loaded = weights::LoadSafetensors(path);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;

  auto b_built = Registry::Instance().Build("detr", TinyConfig());
  ASSERT_TRUE(b_built.has_value()) << b_built.error().message;
  auto b = *b_built;
  auto rep = weights::LoadStateDictInto(*b, *loaded, weights::WeightRemapper{}, /*strict=*/true);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  EXPECT_EQ(rep->loaded, sd.Size());
  EXPECT_TRUE(rep->missing.empty());
  EXPECT_TRUE(rep->unexpected.empty());

  // Same input -> same output after loading identical weights.
  a->eval();
  b->eval();
  torch::NoGradGuard ng;
  auto x = torch::randn({1, 3, 64, 64});
  auto oa = a->Forward(x);
  auto ob = b->Forward(x);
  EXPECT_TRUE(torch::allclose(oa.logits, ob.logits, 1e-4, 1e-4));
  EXPECT_TRUE(torch::allclose(oa.boxes, ob.boxes, 1e-4, 1e-4));

  std::filesystem::remove(path);
}

}  // namespace
}  // namespace detr::models
