// Copyright 2026 detrcpp authors. Apache-2.0.
//
// End-to-end parity: our `dino` model, loading the native IDEA-Research DINO-4scale
// checkpoint, must reproduce the native model's final logits + boxes for a fixed input.
// Golden produced by /tmp/dino_golden.py (native DINO on its pure-PyTorch deform-attn).
// Gated on /tmp/dino_gold/golden.pth + the checkpoint in models/ (skipped when absent).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

#include "detr/models/registry.hpp"
#include "detr/weights/pth.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

TEST(DinoParity, MatchesNativeDino4scale) {
  const std::string wpath = "/home/origo/Desktop/detrcpp/models/dino_checkpoint0033_4scale.pth";
  const std::string gpath = "/tmp/dino_gold/golden.pth";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(gpath)) {
    GTEST_SKIP() << "DINO golden/checkpoint absent (run /tmp/dino_golden.py)";
  }
  RegisterBuiltins();
  auto model = Registry::Instance().Build("dino");
  ASSERT_TRUE(model.has_value()) << model.error().message;
  auto wsd = weights::LoadPth(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(**model, *wsd, (*model)->UpstreamRemapper(), false);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto gsd = *weights::LoadPth(gpath);
  auto input = RawToTorch(gsd.Find("input"));
  auto g_logits = RawToTorch(gsd.Find("logits"));
  auto g_boxes = RawToTorch(gsd.Find("boxes"));

  (*model)->eval();
  torch::NoGradGuard ng;
  auto det = (*model)->Forward(input);
  ASSERT_EQ(det.logits.sizes(), g_logits.sizes());
  ASSERT_EQ(det.boxes.sizes(), g_boxes.sizes());
  const float dl = (det.logits - g_logits).abs().max().item<float>();
  const float db = (det.boxes - g_boxes).abs().max().item<float>();
  std::cout << "logits max|diff| = " << dl << "   boxes max|diff| = " << db << "\n";
  EXPECT_LT(dl, 2e-3F) << "class logits";
  EXPECT_LT(db, 2e-3F) << "boxes";
}

}  // namespace
}  // namespace detr::models
