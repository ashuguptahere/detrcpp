// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ D-FINE HybridEncoder (neck) must reproduce the native HybridEncoder
// outputs for the real backbone features. Gated on the fixtures dumped by
// /tmp/dfine_neck_parity.py (native D-FINE-N neck) + converted weights (skipped when absent).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "detr/models/dfine_encoder.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

// Checks the C++ neck against native outputs for one D-FINE size.
void CheckNeck(DfHybridEncoder neck, int num_levels, const std::string& wpath,
               const std::string& ppath) {
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "D-FINE neck fixtures absent (run /tmp/dfine_neck_parity.py + convert)";
  }
  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*neck, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  for (const auto& mk : rep->missing) std::cout << "  MISSING " << mk << "\n";
  for (const auto& uk : rep->unexpected) std::cout << "  UNEXPECTED " << uk << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  std::vector<torch::Tensor> feats;
  for (int i = 0; i < num_levels; ++i) feats.push_back(RawToTorch(psd.Find("feat" + std::to_string(i))));
  neck->eval();
  torch::NoGradGuard ng;
  auto outs = neck->forward(feats);
  ASSERT_EQ(outs.size(), static_cast<std::size_t>(num_levels));
  for (int i = 0; i < num_levels; ++i) {
    auto ref = RawToTorch(psd.Find("out" + std::to_string(i)));
    ASSERT_EQ(outs[static_cast<std::size_t>(i)].sizes(), ref.sizes());
    const float d = (outs[static_cast<std::size_t>(i)] - ref).abs().max().item<float>();
    std::cout << "out" << i << " max|diff| = " << d << "\n";
    EXPECT_LT(d, 5e-4F) << "out" << i;
  }
}

// D-FINE-N: 2 levels, hidden 128, ff 512, expansion 0.34, depth_mult 0.5, AIFI on level 1.
TEST(DfHybridEncoderParity, MatchesDFineNNeck) {
  CheckNeck(DfHybridEncoder(std::vector<int>{512, 1024}, std::vector<int>{16, 32}, 128, 8, 512, 0.34,
                            0.5, std::vector<int>{1}, 1, 10000.0),
            2, "/tmp/dfine_n_neck.safetensors", "/tmp/dfine_n_neck/parity.safetensors");
}

// D-FINE-L: 3 levels, hidden 256, ff 1024, expansion 1.0, depth_mult 1.0, AIFI on level 2.
TEST(DfHybridEncoderParity, MatchesDFineLNeck) {
  CheckNeck(DfHybridEncoder(std::vector<int>{512, 1024, 2048}, std::vector<int>{8, 16, 32}, 256, 8,
                            1024, 1.0, 1.0, std::vector<int>{2}, 1, 10000.0),
            3, "/tmp/dfine_l_neck.safetensors", "/tmp/dfine_l_neck/parity.safetensors");
}

}  // namespace
}  // namespace detr::models
