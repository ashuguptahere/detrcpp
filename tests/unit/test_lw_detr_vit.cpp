// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ LW-DETR ViT backbone must reproduce the reference LW-DETR-medium
// backbone's feature maps for a fixed input. Gated on the parity fixtures dumped by
// /tmp/lwdetr_dump_parity.py + the converted backbone weights (skipped when absent).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "detr/models/lw_detr_vit.hpp"
#include "detr/models/rf_detr_projector.hpp"
#include "detr/models/rf_detr_real.hpp"
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

// The LW-DETR-large/xlarge multi-scale projector (P3 2x-up + P5 0.5x-down) must
// reproduce the reference's two feature maps from the 4 backbone features.
TEST(LwDetrViTParity, MultiScaleProjectorMatchesLarge) {
  const std::string wpath = "/tmp/lwdetr_large_projector_native.safetensors";
  const std::string ppath = "/tmp/lwdetr_native_large/parity.safetensors";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "large projector fixtures absent";
  }
  auto proj = LwDetrMultiScaleProjector(4, 384, 384, 3, std::vector<double>{2.0, 0.5});
  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*proj, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "msproj loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  std::vector<torch::Tensor> feats;
  for (int i = 0; i < 4; ++i) {
    feats.push_back(RawToTorch(psd.Find("feat" + std::to_string(i))));
  }
  proj->eval();
  torch::NoGradGuard ng;
  auto outs = proj->forward(feats);
  ASSERT_EQ(outs.size(), 2U);
  for (int s = 0; s < 2; ++s) {
    auto ref = RawToTorch(psd.Find("proj" + std::to_string(s)));
    ASSERT_EQ(outs[static_cast<std::size_t>(s)].sizes(), ref.sizes());
    const float d = (outs[static_cast<std::size_t>(s)] - ref).abs().max().item<float>();
    std::cout << "proj" << s << " max|diff| = " << d << "\n";
    EXPECT_LT(d, 5e-3F) << "proj" << s;
  }
}

// LW-DETR-medium config for the shared RfDetrReal family model.
RfDetrRealConfig LwDetrMediumConfig() {
  RfDetrRealConfig c;
  c.name = "lw-detr-medium";
  c.upstream = "https://github.com/Atten4Vis/LW-DETR";
  c.imgsz = 640;
  c.num_classes = 91;
  c.dec_layers = 3;
  c.backbone = RfDetrRealConfig::kLwDetrViT;
  c.vit_embed = 384;
  c.vit_depth = 10;
  c.vit_heads = 12;
  c.patch = 16;
  c.num_windows = 4;  // windows per side
  c.pe_grid = 14;     // pretrain 224/16
  c.projector_batchnorm = true;
  c.out_indices = {2, 4, 5, 9};
  c.window_blocks = {0, 1, 3, 6, 7, 9};
  return c;
}

// LW-DETR-large config: multi-scale (P3+P5), d_model 384, n_points 4.
RfDetrRealConfig LwDetrLargeConfig() {
  RfDetrRealConfig c = LwDetrMediumConfig();
  c.name = "lw-detr-large";
  c.d_model = 384;
  c.n_points = 4;
  c.scale_factors = {2.0, 0.5};
  return c;
}

// LW-DETR-xlarge config: large with a ViT-B backbone (embed 768).
RfDetrRealConfig LwDetrXLargeConfig() {
  RfDetrRealConfig c = LwDetrLargeConfig();
  c.name = "lw-detr-xlarge";
  c.vit_embed = 768;
  return c;
}

// Shared token-aligned end-to-end checker (random input -> a few topk slots reorder).
void CheckEndToEnd(const std::string& wpath, const std::string& ppath, RfDetrRealConfig cfg) {
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "parity fixtures absent";
  }
  auto model = std::make_shared<RfDetrRealImpl>(cfg);
  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*model, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "full loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  for (const auto& mk : rep->missing) std::cout << "  MISSING " << mk << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  auto input = RawToTorch(psd.Find("input"));
  model->eval();
  torch::NoGradGuard ng;
  auto det = model->Forward(input);
  auto rl = RawToTorch(psd.Find("logits")).squeeze(0);
  auto rb = RawToTorch(psd.Find("pred_boxes")).squeeze(0);
  auto cl = det.logits.squeeze(0);
  auto cb = det.boxes.squeeze(0);
  const int Q = static_cast<int>(cb.size(0));
  auto cpp_tok = model->LastTopkIndices().squeeze(0).to(torch::kLong).contiguous();
  auto hf_tok = RawToTorch(psd.Find("topk_idx")).squeeze(0).to(torch::kLong).contiguous();
  auto ca = cpp_tok.accessor<std::int64_t, 1>();
  auto ha = hf_tok.accessor<std::int64_t, 1>();
  std::unordered_map<std::int64_t, int> hf_slot;
  for (int j = 0; j < Q; ++j) hf_slot[ha[j]] = j;
  std::vector<float> boxd, logd;
  int unmatched = 0;
  for (int i = 0; i < Q; ++i) {
    auto it = hf_slot.find(ca[i]);
    if (it == hf_slot.end()) { ++unmatched; continue; }
    boxd.push_back((cb[i] - rb[it->second]).abs().max().item<float>());
    logd.push_back((cl[i] - rl[it->second]).abs().max().item<float>());
  }
  std::sort(boxd.begin(), boxd.end());
  std::sort(logd.begin(), logd.end());
  const auto mid = boxd.size() / 2;
  const auto p90 = boxd.size() * 9 / 10;
  std::cout << "token-aligned " << boxd.size() << "/" << Q << " (set diff " << unmatched
            << ")\nbox p50=" << boxd[mid] << " p90=" << boxd[p90] << " max=" << boxd.back()
            << "\nlogit p50=" << logd[mid] << " p90=" << logd[p90] << " max=" << logd.back() << "\n";
  EXPECT_LE(unmatched, 3);
  EXPECT_LT(boxd[mid], 1e-3F);
  EXPECT_LT(logd[mid], 1e-2F);
  EXPECT_LT(boxd[p90], 3e-3F);
  EXPECT_LT(logd[p90], 1.5e-2F);
}

TEST(LwDetrViTParity, FullLargeEndToEnd) {
  CheckEndToEnd("/tmp/lwdetr_large_full_native.safetensors",
                "/tmp/lwdetr_native_large/parity.safetensors", LwDetrLargeConfig());
}

TEST(LwDetrViTParity, FullXLargeEndToEnd) {
  CheckEndToEnd("/tmp/lwdetr_xlarge_full_native.safetensors",
                "/tmp/lwdetr_native_xlarge/parity.safetensors", LwDetrXLargeConfig());
}

TEST(LwDetrViTParity, FullMediumEndToEnd) {
  const std::string wpath = "/tmp/lwdetr_full.safetensors";
  const std::string ppath = "/tmp/lwdetr_parity/parity.safetensors";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "parity fixtures absent";
  }
  auto model = std::make_shared<RfDetrRealImpl>(LwDetrMediumConfig());
  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*model, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "full loaded " << rep->loaded << " missing " << rep->missing.size() << " unexpected "
            << rep->unexpected.size() << "\n";
  for (const auto& mk : rep->missing) {
    std::cout << "  MISSING " << mk << "\n";
  }
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  auto input = RawToTorch(psd.Find("input"));
  model->eval();
  torch::NoGradGuard ng;
  auto det = model->Forward(input);
  auto rl = RawToTorch(psd.Find("logits")).squeeze(0);
  auto rb = RawToTorch(psd.Find("pred_boxes")).squeeze(0);
  auto cl = det.logits.squeeze(0);
  auto cb = det.boxes.squeeze(0);
  ASSERT_EQ(det.logits.sizes(), RawToTorch(psd.Find("logits")).sizes());

  // Token-aligned parity (random input -> torch.topk tie-break reorders a few of the
  // 300 query slots between backends; align by the encoder token each slot selected).
  const int Q = static_cast<int>(cb.size(0));
  auto cpp_tok = model->LastTopkIndices().squeeze(0).to(torch::kLong).contiguous();
  auto hf_tok = RawToTorch(psd.Find("topk_idx")).squeeze(0).to(torch::kLong).contiguous();
  auto ca = cpp_tok.accessor<std::int64_t, 1>();
  auto ha = hf_tok.accessor<std::int64_t, 1>();
  std::unordered_map<std::int64_t, int> hf_slot;
  for (int j = 0; j < Q; ++j) {
    hf_slot[ha[j]] = j;
  }
  std::vector<float> boxd, logd;
  int unmatched = 0;
  for (int i = 0; i < Q; ++i) {
    auto it = hf_slot.find(ca[i]);
    if (it == hf_slot.end()) {
      ++unmatched;
      continue;
    }
    boxd.push_back((cb[i] - rb[it->second]).abs().max().item<float>());
    logd.push_back((cl[i] - rl[it->second]).abs().max().item<float>());
  }
  std::sort(boxd.begin(), boxd.end());
  std::sort(logd.begin(), logd.end());
  const auto mid = boxd.size() / 2;
  const auto p90 = boxd.size() * 9 / 10;
  std::cout << "token-aligned " << boxd.size() << "/" << Q << " (set diff " << unmatched
            << ")\nbox p50=" << boxd[mid] << " p90=" << boxd[p90] << " max=" << boxd.back()
            << "\nlogit p50=" << logd[mid] << " p90=" << logd[p90] << " max=" << logd.back() << "\n";
  EXPECT_LE(unmatched, 2);
  EXPECT_LT(boxd[mid], 1e-3F);
  EXPECT_LT(logd[mid], 1e-2F);
  EXPECT_LT(boxd[p90], 3e-3F);
  EXPECT_LT(logd[p90], 1.5e-2F);
}

}  // namespace
}  // namespace detr::models
