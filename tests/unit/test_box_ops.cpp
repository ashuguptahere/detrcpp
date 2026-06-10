// Copyright 2026 detrcpp authors. Apache-2.0.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/train/box_ops.hpp"

namespace detr::train {
namespace {

// Random valid xyxy boxes (x1<x2, y1<y2).
torch::Tensor RandBoxes(int n) {
  auto xy = torch::rand({n, 2});
  auto wh = torch::rand({n, 2}) + 0.1;   // strictly positive size
  return torch::cat({xy, xy + wh}, -1);  // [n,4] xyxy
}

// The paired GIoU used by the loss must equal the diagonal of the pairwise GIoU
// used by the matcher — that equivalence is the whole point of the O(M) helper.
TEST(BoxOps, PairedGiouEqualsDiagonalOfPairwise) {
  torch::manual_seed(0);
  auto a = RandBoxes(7);
  auto b = RandBoxes(7);
  auto paired = GeneralizedBoxIouPaired(a, b);         // [7]
  auto diag = GeneralizedBoxIou(a, b).diagonal();      // [7]
  EXPECT_TRUE(torch::allclose(paired, diag, 1e-6, 1e-6));
}

TEST(BoxOps, GiouSanity) {
  // Identical box -> GIoU == 1.
  auto box = torch::tensor({{0.0F, 0.0F, 2.0F, 2.0F}});
  EXPECT_NEAR(GeneralizedBoxIouPaired(box, box).item<float>(), 1.0F, 1e-5);

  // Disjoint boxes -> GIoU in [-1, 0).
  auto a = torch::tensor({{0.0F, 0.0F, 1.0F, 1.0F}});
  auto b = torch::tensor({{3.0F, 3.0F, 4.0F, 4.0F}});
  const float g = GeneralizedBoxIouPaired(a, b).item<float>();
  EXPECT_LT(g, 0.0F);
  EXPECT_GE(g, -1.0F);
}

}  // namespace
}  // namespace detr::train
