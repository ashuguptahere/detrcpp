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
#include "detr/weights/pth.hpp"
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
  const std::string wpath = "/tmp/deimv2_s_backbone.pth";
  const std::string ppath = "/tmp/deimv2_s_bb/parity.pth";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "DEIMv2-S backbone fixtures absent (run /tmp/deimv2s_bb_dump.py + convert)";
  }
  DinoStaConfig cfg;
  cfg.embed_dim = 192;
  cfg.num_heads = 3;
  cfg.conv_inplane = 16;
  cfg.hidden_dim = 192;
  cfg.interaction_indexes = {3, 7, 11};
  auto bb = DinoV3Sta(cfg);
  auto wsd = weights::LoadPth(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*bb, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  for (const auto& mk : rep->missing) std::cout << "  MISSING " << mk << "\n";
  for (const auto& uk : rep->unexpected) std::cout << "  UNEXPECTED " << uk << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadPth(ppath);
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

// DEIMv2-L backbone: Meta DINOv3 ViT-S/16 (embed 384, 6 heads, 12 blocks, GELU MLP,
// 4 storage tokens), interaction [5,8,11], conv_inplane 32, hidden_dim 224.
TEST(DinoV3StaParity, MatchesDeimv2LBackbone) {
  const std::string wpath = "/tmp/deimv2_l_backbone.pth";
  const std::string ppath = "/tmp/deimv2_l_bb/parity.pth";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "DEIMv2-L backbone fixtures absent (run /tmp/deimv2l_bb_dump.py + convert)";
  }
  DinoStaConfig cfg;
  cfg.embed_dim = 384;
  cfg.num_heads = 6;
  cfg.conv_inplane = 32;
  cfg.hidden_dim = 224;
  cfg.interaction_indexes = {5, 8, 11};
  cfg.dinov3_vit = true;
  cfg.ffn_ratio = 4.0;
  cfg.n_storage = 4;
  auto bb = DinoV3Sta(cfg);
  auto wsd = weights::LoadPth(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*bb, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  for (const auto& mk : rep->missing) std::cout << "  MISSING " << mk << "\n";
  for (const auto& uk : rep->unexpected) std::cout << "  UNEXPECTED " << uk << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);
  auto psd = *weights::LoadPth(ppath);
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

// DEIMv2-X backbone: Meta DINOv3 ViT-S+/16 (embed 384, 6 heads, 12 blocks, SwiGLU FFN
// ratio 6, 4 storage tokens), interaction [5,8,11], conv_inplane 64, hidden_dim 256.
TEST(DinoV3StaParity, MatchesDeimv2XBackbone) {
  const std::string wpath = "/tmp/deimv2_x_backbone.pth";
  const std::string ppath = "/tmp/deimv2_x_bb/parity.pth";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "DEIMv2-X backbone fixtures absent (run /tmp/deimv2x_bb_dump.py + convert)";
  }
  DinoStaConfig cfg;
  cfg.embed_dim = 384;
  cfg.num_heads = 6;
  cfg.conv_inplane = 64;
  cfg.hidden_dim = 256;
  cfg.interaction_indexes = {5, 8, 11};
  cfg.dinov3_vit = true;
  cfg.ffn_ratio = 6.0;
  cfg.swiglu = true;
  cfg.n_storage = 4;
  auto bb = DinoV3Sta(cfg);
  auto wsd = weights::LoadPth(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*bb, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  for (const auto& mk : rep->missing) std::cout << "  MISSING " << mk << "\n";
  for (const auto& uk : rep->unexpected) std::cout << "  UNEXPECTED " << uk << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);
  auto psd = *weights::LoadPth(ppath);
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
