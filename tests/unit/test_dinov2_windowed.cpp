// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ DINOv2-windowed backbone (RF-DETR) must reproduce the reference
// RF-DETR-Nano backbone's 4 feature maps for a fixed input. Gated on the parity
// fixtures dumped by /tmp/rfdetr_dump_parity.py + the converted backbone weights
// (skipped in CI when absent).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "detr/models/dinov2_windowed.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

TEST(Dinov2WindowedParity, MatchesRfDetrNanoBackbone) {
  const std::string wpath = "/tmp/rfdetr_backbone.safetensors";
  const std::string ppath = "/tmp/rfdetr_parity/parity.safetensors";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "parity fixtures absent (run /tmp/rfdetr_dump_parity.py + convert)";
  }
  // RF-DETR-Nano: embed 384, depth 12, 6 heads, patch 16, 2 windows, pe grid 24, no
  // registers, features at stages [3,6,9,12], windowed blocks [0,1,2,4,5,7,8,10,11].
  auto bb = Dinov2Windowed(384, 12, 6, 16, 2, 24, 0, std::vector<int>{3, 6, 9, 12},
                           std::vector<int>{0, 1, 2, 4, 5, 7, 8, 10, 11});
  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*bb, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  auto input = RawToTorch(psd.Find("input"));
  bb->eval();
  torch::NoGradGuard ng;
  auto feats = bb->forward(input);
  ASSERT_EQ(feats.size(), 4U);
  for (int i = 0; i < 4; ++i) {
    auto ref = RawToTorch(psd.Find("feat" + std::to_string(i)));
    ASSERT_EQ(feats[static_cast<std::size_t>(i)].sizes(), ref.sizes());
    const float d = (feats[static_cast<std::size_t>(i)] - ref).abs().max().item<float>();
    std::cout << "feat" << i << " max|diff| = " << d << "\n";
    EXPECT_LT(d, 5e-3F) << "feat" << i;
  }
}

}  // namespace
}  // namespace detr::models
