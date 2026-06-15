// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ Swin Transformer backbone must reproduce the native (Lite-DETR /
// mmdet-style) Swin-T backbone's four stage feature maps for a fixed input, loading the
// official microsoft Swin-T pretrain. Gated on /tmp/swin_dump.py fixtures (skipped when
// absent). The detection per-stage norms (norm0..3) are default-init in both sides.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "detr/models/swin.hpp"
#include "detr/weights/pth.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

TEST(SwinParity, MatchesSwinTBackbone) {
  const std::string wpath = "/tmp/swin_t_backbone.pth";
  const std::string ppath = "/tmp/swin_t_bb/parity.pth";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "Swin-T backbone fixtures absent (run /tmp/swin_dump.py)";
  }
  auto bb = SwinTransformer(SwinConfigFor("swin_T_224_1k"));
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
  ASSERT_EQ(outs.size(), 4U);
  for (int i = 0; i < 4; ++i) {
    auto ref = RawToTorch(psd.Find("s" + std::to_string(i)));
    ASSERT_EQ(outs[static_cast<std::size_t>(i)].sizes(), ref.sizes());
    const float d = (outs[static_cast<std::size_t>(i)] - ref).abs().max().item<float>();
    std::cout << "s" << i << " max|diff| = " << d << "\n";
    EXPECT_LT(d, 2e-3F) << "stage " << i;
  }
}

}  // namespace
}  // namespace detr::models
