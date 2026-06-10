// Copyright 2026 detrcpp authors. Apache-2.0.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/models/frozen_batchnorm.hpp"

namespace detr::models {

// FrozenBatchNorm2d must be numerically identical to a standard BatchNorm2d in
// eval mode for the same affine params + running stats. This is the guarantee
// that swapping the ResNet backbone to FrozenBN preserves the exact eval-time
// behavior (and hence the validated COCO mAP) of the weight-loaded models.
TEST(FrozenBatchNorm2d, MatchesBatchNormEval) {
  torch::manual_seed(0);
  const std::int64_t c = 8;

  auto bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c));
  {
    torch::NoGradGuard ng;
    bn->weight.normal_();
    bn->bias.normal_();
    bn->running_mean.normal_();
    bn->running_var.uniform_(0.5, 2.0);  // strictly positive variance
  }
  bn->eval();

  FrozenBatchNorm2d fbn(c);
  {
    torch::NoGradGuard ng;
    fbn->weight.copy_(bn->weight);
    fbn->bias.copy_(bn->bias);
    fbn->running_mean.copy_(bn->running_mean);
    fbn->running_var.copy_(bn->running_var);
  }

  auto x = torch::randn({2, c, 5, 7});
  EXPECT_TRUE(torch::allclose(bn->forward(x), fbn->forward(x), /*rtol=*/1e-5, /*atol=*/1e-6));
}

// The frozen params must not collect gradients (they are buffers, not params).
TEST(FrozenBatchNorm2d, ParamsAreNotTrainable) {
  FrozenBatchNorm2d fbn(4);
  EXPECT_TRUE(fbn->parameters().empty());
  EXPECT_EQ(fbn->buffers().size(), 4U);
}

}  // namespace detr::models
