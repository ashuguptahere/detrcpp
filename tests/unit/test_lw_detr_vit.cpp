// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ LW-DETR ViT backbone must reproduce the reference LW-DETR-medium
// backbone's feature maps for a fixed input. Gated on the parity fixtures dumped by
// /tmp/lwdetr_dump_parity.py + the converted backbone weights (skipped when absent).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "detr/models/lw_detr_vit.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

TEST(LwDetrViTParity, MatchesMediumBackbone) {
  const std::string wpath = "/tmp/lwdetr_backbone.safetensors";
  const std::string ppath = "/tmp/lwdetr_parity/parity.safetensors";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "parity fixtures absent (run /tmp/lwdetr_dump_parity.py + convert)";
  }
  // LW-DETR-medium: ViT-S (embed 384, 10 layers, 12 heads, patch 16), 4x4 windows,
  // pretrain pos grid 14, features after layers [2,4,5,9], windowed [0,1,3,6,7,9].
  auto bb = LwDetrViT(384, 10, 12, 16, 4, 14, std::vector<int>{2, 4, 5, 9},
                      std::vector<int>{0, 1, 3, 6, 7, 9});
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
