// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ HGNetv2 backbone must reproduce the native D-FINE HGNetv2 feature
// maps for a fixed input. Gated on the fixtures dumped by /tmp/dfine_backbone_parity.py
// (native B0 backbone from dfine_n_coco.pth) + the converted weights (skipped when absent).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "detr/models/hgnetv2.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

// Checks a C++ HGNetv2 against native feature maps for one D-FINE size.
void CheckBackbone(const std::string& variant, bool use_lab, const std::vector<int>& return_idx,
                   const std::string& wpath, const std::string& ppath) {
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "D-FINE backbone fixtures absent (run /tmp/dfine_backbone_parity.py + convert)";
  }
  auto bb = HgNetV2(variant, use_lab, return_idx);
  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*bb, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << variant << " loaded " << rep->loaded << " missing " << rep->missing.size()
            << " unexpected " << rep->unexpected.size() << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  auto input = RawToTorch(psd.Find("input"));
  bb->eval();
  torch::NoGradGuard ng;
  auto feats = bb->forward(input);
  ASSERT_EQ(feats.size(), return_idx.size());
  for (std::size_t i = 0; i < feats.size(); ++i) {
    auto ref = RawToTorch(psd.Find("feat" + std::to_string(i)));
    ASSERT_EQ(feats[i].sizes(), ref.sizes());
    const float d = (feats[i] - ref).abs().max().item<float>();
    std::cout << "feat" << i << " max|diff| = " << d << "\n";
    EXPECT_LT(d, 5e-4F) << "feat" << i;
  }
}

// D-FINE-N: HGNetv2-B0, use_lab=True, return_idx [2,3] (strides 16/32).
TEST(HgNetV2Parity, MatchesDFineNBackbone) {
  CheckBackbone("B0", true, {2, 3}, "/tmp/dfine_n_backbone.safetensors", "/tmp/dfine_n_bb/parity.safetensors");
}

// D-FINE-L: HGNetv2-B4, use_lab=False, return_idx [1,2,3] (strides 8/16/32).
TEST(HgNetV2Parity, MatchesDFineLBackbone) {
  CheckBackbone("B4", false, {1, 2, 3}, "/tmp/dfine_l_backbone.safetensors", "/tmp/dfine_l_bb/parity.safetensors");
}

// DEIMv2-Atto: micro 3-stage HGNetv2, use_lab=True, return_idx [2] (single feature, stride 16).
TEST(HgNetV2Parity, MatchesDeimv2AttoBackbone) {
  CheckBackbone("Atto", true, {2}, "/tmp/deimv2_atto_backbone.safetensors", "/tmp/deimv2_atto_bb/parity.safetensors");
}

}  // namespace
}  // namespace detr::models
