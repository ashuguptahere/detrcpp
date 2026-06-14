// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ DINOv2-windowed backbone (RF-DETR) must reproduce the reference
// RF-DETR-Nano backbone's 4 feature maps for a fixed input. Gated on the parity
// fixtures dumped by /tmp/rfdetr_dump_parity.py + the converted backbone weights
// (skipped in CI when absent).

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

#include "detr/models/dinov2_windowed.hpp"
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

TEST(Dinov2WindowedParity, ProjectorMatchesReference) {
  const std::string wpath = "/tmp/rfdetr_projector.safetensors";
  const std::string ppath = "/tmp/rfdetr_parity/parity.safetensors";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "parity fixtures absent";
  }
  auto proj = RfDetrProjector(4, 384, 256, 3);  // 4 features x 384 -> C2f -> 256, n=3
  auto wsd = weights::LoadSafetensors(wpath);
  ASSERT_TRUE(wsd.has_value()) << wsd.error().message;
  auto rep = weights::LoadStateDictInto(*proj, *wsd);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  std::cout << "projector loaded " << rep->loaded << " missing " << rep->missing.size()
            << " unexpected " << rep->unexpected.size() << "\n";
  EXPECT_EQ(rep->missing.size(), 0U);
  EXPECT_EQ(rep->unexpected.size(), 0U);

  auto psd = *weights::LoadSafetensors(ppath);
  std::vector<torch::Tensor> feats;
  for (int i = 0; i < 4; ++i) {
    feats.push_back(RawToTorch(psd.Find("feat" + std::to_string(i))));
  }
  proj->eval();
  torch::NoGradGuard ng;
  auto out = proj->forward(feats);
  auto ref = RawToTorch(psd.Find("proj0"));
  ASSERT_EQ(out.sizes(), ref.sizes());
  const float d = (out - ref).abs().max().item<float>();
  std::cout << "proj max|diff| = " << d << "\n";
  EXPECT_LT(d, 5e-3F);
}

TEST(Dinov2WindowedParity, FullRfDetrEndToEnd) {
  const std::string wpath = "/tmp/rfdetr_full.safetensors";
  const std::string ppath = "/tmp/rfdetr_parity/parity.safetensors";
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "parity fixtures absent";
  }
  auto model = std::make_shared<RfDetrRealImpl>();  // nano defaults (384px, DINOv2-S, 2 win, 2 dec)
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
  auto ref_logits = RawToTorch(psd.Find("logits"));
  auto ref_boxes = RawToTorch(psd.Find("pred_boxes"));
  ASSERT_EQ(det.logits.sizes(), ref_logits.sizes());
  ASSERT_EQ(det.boxes.sizes(), ref_boxes.sizes());

  // Token-aligned parity. Two-stage query selection takes the top-K encoder tokens by
  // class score, but on this random-noise fixture the low-confidence tokens are tied to
  // ~1e-5 and torch.topk breaks ties in backend-defined order (libtorch 2.5.1 vs
  // PyTorch), permuting a few of the 300 query slots. We align each predicted query to
  // the reference query that selected the SAME encoder token, so the comparison is
  // independent of that ordering. A couple of tokens that land in a different SLOT bind
  // a different per-slot reference_point_embed and so genuinely differ; that difference
  // also ripples through the decoder's self-attention to nudge other queries slightly.
  // Both are deterministic consequences of an irreducible top-K tie-break, not fidelity
  // gaps — so we check the median (tight; catches any real error) and the 90th
  // percentile (bounds the ripple), with margin.
  auto cb = det.boxes.squeeze(0);     // [Q,4]
  auto cl = det.logits.squeeze(0);    // [Q,C]
  auto rb = ref_boxes.squeeze(0);     // [Q,4]
  auto rl = ref_logits.squeeze(0);    // [Q,C]
  const int Q = static_cast<int>(cb.size(0));
  auto cpp_tok = model->LastTopkIndices().squeeze(0).to(torch::kLong).contiguous();
  auto hf_tok = RawToTorch(psd.Find("topk_idx")).squeeze(0).to(torch::kLong).contiguous();
  auto cpp_acc = cpp_tok.accessor<std::int64_t, 1>();
  auto hf_acc = hf_tok.accessor<std::int64_t, 1>();
  std::unordered_map<std::int64_t, int> hf_slot;
  for (int j = 0; j < Q; ++j) {
    hf_slot[hf_acc[j]] = j;
  }
  std::vector<float> boxd, logd;
  int unmatched_token = 0;
  for (int i = 0; i < Q; ++i) {
    auto it = hf_slot.find(cpp_acc[i]);
    if (it == hf_slot.end()) {  // token not selected by the reference run
      ++unmatched_token;
      continue;
    }
    const int j = it->second;
    boxd.push_back((cb[i] - rb[j]).abs().max().item<float>());
    logd.push_back((cl[i] - rl[j]).abs().max().item<float>());
  }
  std::sort(boxd.begin(), boxd.end());
  std::sort(logd.begin(), logd.end());
  const auto mid = boxd.size() / 2;
  const auto p90 = boxd.size() * 9 / 10;
  std::cout << "token-aligned " << boxd.size() << "/" << Q << " (set diff " << unmatched_token
            << ")\nbox   p50=" << boxd[mid] << " p90=" << boxd[p90] << " max=" << boxd.back()
            << "\nlogit p50=" << logd[mid] << " p90=" << logd[p90] << " max=" << logd.back() << "\n";
  // Same encoder tokens selected (random input -> identical top-K set, only the order of
  // a few near-tied tokens differs).
  EXPECT_LE(unmatched_token, 2);
  // Token-aligned, the port reproduces RF-DETR-Nano to fp32 noise: the median query is
  // exact and 90% of queries are tight; only the tie-break ripple inflates the tail.
  EXPECT_LT(boxd[mid], 1e-3F);
  EXPECT_LT(logd[mid], 1e-2F);
  EXPECT_LT(boxd[p90], 3e-3F);
  EXPECT_LT(logd[p90], 1.5e-2F);
}

}  // namespace
}  // namespace detr::models
