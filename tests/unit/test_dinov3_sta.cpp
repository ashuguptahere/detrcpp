// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ DINOv3-STA backbone (RoPE ViT-Tiny + Spatial-Tuning Adapter) must
// reproduce the native DEIMv2-S backbone's three fused feature maps for a fixed input.
// Gated on /tmp/deimv2s_bb_dump.py fixtures + converted weights (skipped when absent).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "detr/models/dinov3_sta.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

// DEIMv2-S backbone: vit_tiny (embed 192, 3 heads, 12 blocks), interaction [3,7,11],
// conv_inplane 16, hidden_dim 192. Three scales at strides 8/16/32.
TEST(DinoV3StaParity, MatchesDeimv2SBackbone) {
  const std::string wpath = "/tmp/deimv2_s_backbone.safetensors";
  const std::string ppath = "/tmp/deimv2_s_bb/parity.safetensors";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "DEIMv2-S backbone fixtures absent (run /tmp/deimv2s_bb_dump.py + convert)";
  }
  auto bb = DinoV3Sta(192, 3, 12, 16, std::vector<int>{3, 7, 11}, 16, 192);
  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*bb, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  for (const auto& mk : rep->missing) std::cout << "  MISSING " << mk << "\n";
  for (const auto& uk : rep->unexpected) std::cout << "  UNEXPECTED " << uk << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  auto input = RawToTorch(psd.Find("input"));
  bb->eval();
  torch::NoGradGuard ng;
  auto outs = bb->forward(input);
  ASSERT_EQ(outs.size(), 3U);
  const char* names[] = {"c2", "c3", "c4"};
  for (int i = 0; i < 3; ++i) {
    auto ref = RawToTorch(psd.Find(names[i]));
    ASSERT_EQ(outs[static_cast<std::size_t>(i)].sizes(), ref.sizes());
    const float d = (outs[static_cast<std::size_t>(i)] - ref).abs().max().item<float>();
    std::cout << names[i] << " max|diff| = " << d << "\n";
    EXPECT_LT(d, 2e-3F) << names[i];
  }
}

}  // namespace
}  // namespace detr::models
