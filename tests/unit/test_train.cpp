// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Verifies DETR's matching + loss: the Hungarian matcher pairs each ground-truth
// box with the closest query, and the SetCriterion produces finite, fully
// differentiable losses. Also checks box ops (cxcywh<->xyxy, GIoU).

#include "detr/train/criterion.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/models/detr.hpp"
#include "detr/models/model.hpp"
#include "detr/models/registry.hpp"
#include "detr/train/box_ops.hpp"
#include "detr/train/checkpoint.hpp"
#include "detr/train/ema.hpp"
#include "detr/train/matcher.hpp"
#include "detr/train/target.hpp"
#include "detr/train/trainer.hpp"
#include "detr/weights/state_dict.hpp"

namespace detr::train {
namespace {

TEST(BoxOps, CxcywhXyxyRoundTrip) {
  auto boxes = torch::tensor({{0.5F, 0.5F, 0.2F, 0.4F}, {0.25F, 0.75F, 0.1F, 0.1F}});
  auto back = BoxXyxyToCxcywh(BoxCxcywhToXyxy(boxes));
  EXPECT_TRUE(torch::allclose(boxes, back, 1e-5, 1e-5));
}

TEST(BoxOps, GiouIdenticalBoxesIsOne) {
  auto a = torch::tensor({{0.0F, 0.0F, 1.0F, 1.0F}});
  auto giou = GeneralizedBoxIou(a, a);
  EXPECT_NEAR(giou.item<float>(), 1.0F, 1e-5);
}

// Builds a 1-image batch: 4 queries, 2 targets. Queries 1 and 3 sit exactly on
// targets 0 and 1, and are confident about the right classes.
models::Detections MakeOutputs(bool requires_grad) {
  auto logits = torch::full({1, 4, 4}, -2.0F);  // num_classes=3 -> 4 logits
  logits[0][1][0] = 5.0F;                        // query1 -> class 0
  logits[0][3][2] = 5.0F;                        // query3 -> class 2
  auto boxes = torch::tensor({{{0.10F, 0.10F, 0.05F, 0.05F},
                               {0.25F, 0.25F, 0.20F, 0.20F},
                               {0.50F, 0.50F, 0.05F, 0.05F},
                               {0.75F, 0.75F, 0.20F, 0.20F}}});
  if (requires_grad) {
    logits.set_requires_grad(true);
    boxes.set_requires_grad(true);
  }
  models::Detections out;
  out.logits = logits;
  out.boxes = boxes;
  return out;
}

TargetBatch MakeTargets() {
  Target t;
  t.labels = torch::tensor({0L, 2L});
  t.boxes = torch::tensor({{0.25F, 0.25F, 0.20F, 0.20F}, {0.75F, 0.75F, 0.20F, 0.20F}});
  return {t};
}

TEST(Matcher, MatchesClosestQueries) {
  auto out = MakeOutputs(false);
  auto targets = MakeTargets();
  auto matches = HungarianMatch(out, targets);
  ASSERT_EQ(matches.size(), 1U);
  const auto& [src, tgt] = matches[0];
  ASSERT_EQ(src.numel(), 2);
  ASSERT_EQ(tgt.numel(), 2);
  // Build a query->target map and check 1->0 and 3->1.
  std::map<int64_t, int64_t> q2t;
  for (int i = 0; i < 2; ++i) {
    q2t[src[i].item<int64_t>()] = tgt[i].item<int64_t>();
  }
  EXPECT_EQ(q2t[1], 0);
  EXPECT_EQ(q2t[3], 1);
}

TEST(Criterion, ProducesFiniteDifferentiableLoss) {
  auto out = MakeOutputs(/*requires_grad=*/true);
  auto targets = MakeTargets();
  auto matches = HungarianMatch(out, targets);

  SetCriterion criterion(/*num_classes=*/3, LossWeights{});
  auto losses = criterion.Compute(out, targets, matches);

  EXPECT_TRUE(std::isfinite(losses.total.item<float>()));
  EXPECT_GE(losses.loss_ce.item<float>(), 0.0F);
  EXPECT_GE(losses.loss_bbox.item<float>(), 0.0F);
  EXPECT_GE(losses.total.item<float>(), 0.0F);

  // Gradients flow back to the predictions.
  losses.total.backward();
  ASSERT_TRUE(out.logits.grad().defined());
  ASSERT_TRUE(out.boxes.grad().defined());
  EXPECT_GT(out.logits.grad().abs().sum().item<float>(), 0.0F);
}

// ---- Trainer (full step: model + matcher + loss + AdamW + EMA) ----

YAML::Node TinyDetrConfig() {
  YAML::Node c;
  c["hidden_dim"] = 32;
  c["nheads"] = 4;
  c["enc_layers"] = 1;
  c["dec_layers"] = 1;
  c["dim_feedforward"] = 64;
  c["num_queries"] = 10;
  c["num_classes"] = 4;
  c["backbone_width"] = 8;
  return c;
}

// A fixed synthetic batch: 2 images, 2 objects each, classes in [0,3].
std::pair<torch::Tensor, TargetBatch> SyntheticBatch() {
  auto images = torch::randn({2, 3, 64, 64});
  TargetBatch targets;
  for (int b = 0; b < 2; ++b) {
    Target t;
    t.labels = torch::tensor({static_cast<int64_t>(b), static_cast<int64_t>(b + 2)});
    t.boxes = torch::tensor({{0.3F, 0.3F, 0.2F, 0.2F}, {0.7F, 0.6F, 0.25F, 0.3F}});
    targets.push_back(t);
  }
  return {images, targets};
}

class TrainerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    models::RegisterBuiltins();
    torch::manual_seed(0);
  }
};

TEST_F(TrainerTest, OverfitsTinyBatch) {
  auto built = models::Registry::Instance().Build("detr", TinyDetrConfig());
  ASSERT_TRUE(built.has_value()) << built.error().message;

  TrainConfig cfg;
  cfg.lr = 2e-3;
  cfg.seed = 0;
  cfg.grad_clip = 0.1;
  Trainer trainer(*built, cfg);

  torch::manual_seed(123);
  auto [images, targets] = SyntheticBatch();

  const float first = trainer.TrainStep(images, targets);
  float last = first;
  for (int i = 0; i < 200; ++i) {
    last = trainer.TrainStep(images, targets);
  }
  EXPECT_TRUE(std::isfinite(last));
  // Overfitting a fixed batch must drive the loss well down.
  EXPECT_LT(last, first * 0.7F) << "first=" << first << " last=" << last;
}

TEST_F(TrainerTest, CheckpointSaveLoadRestoresStateAndWeights) {
  auto a = models::Registry::Instance().Build("detr", TinyDetrConfig());
  ASSERT_TRUE(a.has_value());
  TrainConfig cfg;
  cfg.lr = 1e-3;
  Trainer ta(*a, cfg);

  auto [images, targets] = SyntheticBatch();
  for (int i = 0; i < 5; ++i) {
    ta.TrainStep(images, targets);
  }

  const auto dir = std::filesystem::temp_directory_path() / "detr_ckpt_test";
  CheckpointMgr ck(dir);
  TrainState st;
  st.epoch = 2;
  st.global_step = 7;
  st.seed = 123;
  st.best_metric = 0.42;
  ASSERT_TRUE(ck.Save("last", ta.Model(), ta.Ema(), ta.Optimizer(), st).has_value());

  // Fresh trainer with different random init.
  torch::manual_seed(999);
  auto b = models::Registry::Instance().Build("detr", TinyDetrConfig());
  ASSERT_TRUE(b.has_value());
  Trainer tb(*b, cfg);

  auto loaded = ck.Load("last", tb.Model(), tb.Ema(), tb.Optimizer());
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  EXPECT_EQ(loaded->epoch, 2);
  EXPECT_EQ(loaded->global_step, 7);
  EXPECT_EQ(loaded->seed, 123U);
  EXPECT_NEAR(loaded->best_metric, 0.42, 1e-9);

  // Raw model weights restored: a named parameter matches.
  auto pa = ta.Model().named_parameters();
  auto pb = tb.Model().named_parameters();
  EXPECT_TRUE(torch::allclose(pa["class_embed.weight"], pb["class_embed.weight"]));

  // EMA shadow restored: compare one tensor's bytes.
  auto sa = ta.Ema().State();
  auto sb = tb.Ema().State();
  const auto* ka = sa.Find("class_embed.weight");
  const auto* kb = sb.Find("class_embed.weight");
  ASSERT_NE(ka, nullptr);
  ASSERT_NE(kb, nullptr);
  EXPECT_EQ(ka->data, kb->data);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

}  // namespace
}  // namespace detr::train
