// Copyright 2026 detrcpp authors. Apache-2.0.
//
// detr-r50 (ResNet-50 backbone) end-to-end smoke: registers, builds from config,
// forward shapes, a single real training step, and a weight round-trip. The full
// trainer/eval/predict/ONNX paths are shared with `detr` and covered elsewhere;
// this confirms the new model slots into all of them.

#include <filesystem>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/models/registry.hpp"
#include "detr/train/target.hpp"
#include "detr/train/trainer.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

YAML::Node R50Tiny() {
  YAML::Node c;
  c["hidden_dim"] = 32;
  c["nheads"] = 4;
  c["enc_layers"] = 1;
  c["dec_layers"] = 1;
  c["dim_feedforward"] = 64;
  c["num_queries"] = 6;
  c["num_classes"] = 4;
  c["imgsz"] = 64;  // ResNet-50 downsamples 32x -> 2x2 feature
  return c;
}

class DetrR50Test : public ::testing::Test {
 protected:
  void SetUp() override { RegisterBuiltins(); }
};

TEST_F(DetrR50Test, RegistersAndListsBothModels) {
  EXPECT_TRUE(Registry::Instance().Contains("detr"));
  EXPECT_TRUE(Registry::Instance().Contains("detr-r50"));
}

TEST_F(DetrR50Test, ForwardShapes) {
  auto built = Registry::Instance().Build("detr-r50", R50Tiny());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  auto model = *built;
  model->eval();
  torch::NoGradGuard ng;
  auto out = model->Forward(torch::randn({1, 3, 64, 64}));
  EXPECT_EQ(out.logits.sizes(), (std::vector<std::int64_t>{1, 6, 5}));  // C+1
  EXPECT_EQ(out.boxes.sizes(), (std::vector<std::int64_t>{1, 6, 4}));
  EXPECT_GE(out.boxes.min().item<float>(), 0.0F);
  EXPECT_LE(out.boxes.max().item<float>(), 1.0F);
}

TEST_F(DetrR50Test, RunsATrainingStep) {
  auto built = Registry::Instance().Build("detr-r50", R50Tiny());
  ASSERT_TRUE(built.has_value());
  train::TrainConfig tc;
  tc.lr = 1e-4;
  train::Trainer trainer(*built, tc);

  auto images = torch::randn({1, 3, 64, 64});
  train::Target t;
  t.labels = torch::tensor({0L, 2L});
  t.boxes = torch::tensor({{0.3F, 0.3F, 0.2F, 0.2F}, {0.7F, 0.6F, 0.25F, 0.3F}});
  const float loss = trainer.TrainStep(images, {t});
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_GE(loss, 0.0F);
}

TEST_F(DetrR50Test, WeightsRoundTrip) {
  auto a = *Registry::Instance().Build("detr-r50", R50Tiny());
  auto sd = weights::StateDictFromModule(*a);
  // ResNet-50 has many tensors (stem + 16 bottlenecks) + the transformer head.
  EXPECT_GT(sd.Size(), 100U);

  const auto path = std::filesystem::temp_directory_path() / "detr_r50.safetensors";
  ASSERT_TRUE(weights::SaveSafetensors(path, sd).has_value());
  auto loaded = weights::LoadSafetensors(path);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;

  auto b = *Registry::Instance().Build("detr-r50", R50Tiny());
  auto rep = weights::LoadStateDictInto(*b, *loaded, weights::WeightRemapper{}, /*strict=*/true);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  EXPECT_EQ(rep->loaded, sd.Size());
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace detr::models
