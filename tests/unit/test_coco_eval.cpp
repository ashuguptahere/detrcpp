// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/eval/coco_eval.hpp"

#include <vector>

#include <gtest/gtest.h>

namespace detr::eval {
namespace {

GtBox Gt(int cat, float x, float y, float w, float h, bool crowd = false) {
  return GtBox{cat, x, y, w, h, crowd};
}
DtBox Dt(int cat, float x, float y, float w, float h, float score) {
  return DtBox{cat, x, y, w, h, score};
}

TEST(CocoEval, PerfectDetectionsGiveApOne) {
  std::vector<EvalImage> imgs(2);
  // Two images, each one object (area 40x40 = 1600 -> medium), perfectly hit.
  imgs[0].gts = {Gt(1, 10, 10, 40, 40)};
  imgs[0].dts = {Dt(1, 10, 10, 40, 40, 0.9F)};
  imgs[1].gts = {Gt(1, 20, 20, 40, 40)};
  imgs[1].dts = {Dt(1, 20, 20, 40, 40, 0.8F)};

  auto m = CocoEvaluate(imgs, {1});
  EXPECT_NEAR(m.ap, 1.0, 1e-9);
  EXPECT_NEAR(m.ap50, 1.0, 1e-9);
  EXPECT_NEAR(m.ap75, 1.0, 1e-9);
  EXPECT_NEAR(m.ar100, 1.0, 1e-9);
  EXPECT_NEAR(m.ap_medium, 1.0, 1e-9);
  EXPECT_DOUBLE_EQ(m.ap_small, -1.0);  // no small gt
  EXPECT_DOUBLE_EQ(m.ap_large, -1.0);  // no large gt
}

TEST(CocoEval, MissedGroundTruthHalvesRecall) {
  // One image, two ground truths, only one detected -> recall caps at 0.5.
  // COCO 101-pt AP = (#recThrs <= 0.5)/101 = 51/101.
  std::vector<EvalImage> imgs(1);
  imgs[0].gts = {Gt(1, 10, 10, 40, 40), Gt(1, 100, 100, 40, 40)};
  imgs[0].dts = {Dt(1, 10, 10, 40, 40, 0.9F)};

  auto m = CocoEvaluate(imgs, {1});
  EXPECT_NEAR(m.ap, 51.0 / 101.0, 1e-9);
  EXPECT_NEAR(m.ar100, 0.5, 1e-9);
}

TEST(CocoEval, AreaBreakdownIsolatesSmallObjects) {
  // One small object (10x10 area 100 < 32^2) perfectly detected.
  std::vector<EvalImage> imgs(1);
  imgs[0].gts = {Gt(1, 5, 5, 10, 10)};
  imgs[0].dts = {Dt(1, 5, 5, 10, 10, 0.9F)};

  auto m = CocoEvaluate(imgs, {1});
  EXPECT_NEAR(m.ap_small, 1.0, 1e-9);
  EXPECT_DOUBLE_EQ(m.ap_medium, -1.0);
  EXPECT_DOUBLE_EQ(m.ap_large, -1.0);
}

TEST(CocoEval, NoDetectionsGiveZero) {
  std::vector<EvalImage> imgs(1);
  imgs[0].gts = {Gt(1, 10, 10, 40, 40)};
  imgs[0].dts = {};
  auto m = CocoEvaluate(imgs, {1});
  EXPECT_NEAR(m.ap, 0.0, 1e-9);
  EXPECT_NEAR(m.ar100, 0.0, 1e-9);
}

TEST(CocoEval, ExtraFalsePositiveBelowDoesNotHurtAp) {
  // One gt, the correct detection scored higher than a stray false positive ->
  // recall reaches 1.0 with precision 1.0 at that point; AP stays 1.0.
  std::vector<EvalImage> imgs(1);
  imgs[0].gts = {Gt(1, 10, 10, 40, 40)};
  imgs[0].dts = {Dt(1, 10, 10, 40, 40, 0.9F), Dt(1, 300, 300, 20, 20, 0.4F)};
  auto m = CocoEvaluate(imgs, {1});
  EXPECT_NEAR(m.ap, 1.0, 1e-9);
}

}  // namespace
}  // namespace detr::eval
