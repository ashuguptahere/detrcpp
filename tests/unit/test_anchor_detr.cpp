// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ anchor-detr-dc5 (ResNet-50-DC5 backbone + RCDA transformer + anchor
// queries) must reproduce the native AnchorDETR's pred_logits / pred_boxes for a fixed
// input. Gated on /tmp/anchordetr_parity.py fixtures + converted weights (skipped when
// absent). Queries are in a fixed order (no top-k selection), so comparison is direct.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "detr/models/anchor_detr.hpp"
#include "detr/models/detr.hpp"
#include "detr/models/registry.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

TEST(AnchorDetrParity, MatchesNativeR50Dc5) {
  const std::string wpath = "/tmp/anchordetr_r50dc5.safetensors";
  const std::string ppath = "/tmp/anchordetr_par/parity.safetensors";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "AnchorDETR fixtures absent (run /tmp/anchordetr_parity.py + convert)";
  }
  RegisterBuiltins();
  auto model_r = Registry::Instance().Build("anchor-detr-dc5", {});
  ASSERT_TRUE(model_r.has_value()) << model_r.error().message;
  auto model = *model_r;

  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*model, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  for (const auto& mk : rep->missing) std::cout << "  MISSING " << mk << "\n";
  for (const auto& uk : rep->unexpected) std::cout << "  UNEXPECTED " << uk << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  auto input = RawToTorch(psd.Find("input"));
  model->eval();
  torch::NoGradGuard ng;
  auto det = model->Forward(input);
  auto rl = RawToTorch(psd.Find("pred_logits"));
  auto rb = RawToTorch(psd.Find("pred_boxes"));
  ASSERT_EQ(det.logits.sizes(), rl.sizes());
  ASSERT_EQ(det.boxes.sizes(), rb.sizes());
  const float dl = (det.logits - rl).abs().max().item<float>();
  const float db = (det.boxes - rb).abs().max().item<float>();
  std::cout << "logit max|diff| = " << dl << "  box max|diff| = " << db << "\n";
  EXPECT_LT(dl, 2e-3F);
  EXPECT_LT(db, 1e-3F);
}

}  // namespace
}  // namespace detr::models
