// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity: the C++ D-FINE decoder + heads (DFINETransformer, FDR) must reproduce the
// native decoder's logits/boxes for the real neck features. Gated on the fixtures from
// /tmp/dfine_decoder_parity.py (native D-FINE-N decoder) + converted weights.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "detr/models/dfine_decoder.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

torch::Tensor RawToTorch(const weights::RawTensor* rt) {
  std::vector<std::int64_t> shape(rt->shape.begin(), rt->shape.end());
  return torch::from_blob(const_cast<std::byte*>(rt->data.data()), shape, torch::kFloat32).clone();
}

void CheckDecoder(const DfTransformerConfig& cfg, int num_levels, const std::string& wpath,
                  const std::string& ppath) {
  if (!std::filesystem::exists(wpath) || !std::filesystem::exists(ppath)) {
    GTEST_SKIP() << "D-FINE decoder fixtures absent (run /tmp/dfine_decoder_parity.py + convert)";
  }
  auto model = DFINETransformer(cfg);
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
  std::vector<torch::Tensor> feats;
  for (int i = 0; i < num_levels; ++i) feats.push_back(RawToTorch(psd.Find("feat" + std::to_string(i))));
  model->eval();
  torch::NoGradGuard ng;
  auto [logits, boxes] = model->forward(feats);
  auto rl = RawToTorch(psd.Find("pred_logits")).squeeze(0);
  auto rb = RawToTorch(psd.Find("pred_boxes")).squeeze(0);
  auto cl = logits.squeeze(0);
  auto cb = boxes.squeeze(0);

  // Token-aligned parity (torch.topk tie-breaks reorder a few of the 300 slots).
  const int Q = static_cast<int>(cb.size(0));
  auto cpp_tok = model->LastTopkIndices().squeeze(0).to(torch::kLong).contiguous();
  auto hf_tok = RawToTorch(psd.Find("topk_idx")).squeeze(0).to(torch::kLong).contiguous();
  auto ca = cpp_tok.accessor<std::int64_t, 1>();
  auto ha = hf_tok.accessor<std::int64_t, 1>();
  std::unordered_map<std::int64_t, int> slot;
  for (int j = 0; j < Q; ++j) slot[ha[j]] = j;
  std::vector<float> boxd, logd;
  int unmatched = 0;
  for (int i = 0; i < Q; ++i) {
    auto it = slot.find(ca[i]);
    if (it == slot.end()) { ++unmatched; continue; }
    boxd.push_back((cb[i] - rb[it->second]).abs().max().item<float>());
    logd.push_back((cl[i] - rl[it->second]).abs().max().item<float>());
  }
  std::sort(boxd.begin(), boxd.end());
  std::sort(logd.begin(), logd.end());
  const auto mid = boxd.size() / 2, p90 = boxd.size() * 9 / 10;
  std::cout << "token-aligned " << boxd.size() << "/" << Q << " (set diff " << unmatched
            << ")\nbox p50=" << boxd[mid] << " p90=" << boxd[p90] << " max=" << boxd.back()
            << "\nlogit p50=" << logd[mid] << " p90=" << logd[p90] << " max=" << logd.back() << "\n";
  EXPECT_LE(unmatched, 3);
  EXPECT_LT(boxd[mid], 1e-3F);
  EXPECT_LT(logd[mid], 1e-2F);
  EXPECT_LT(boxd[p90], 3e-3F);
  EXPECT_LT(logd[p90], 1.5e-2F);
}

// D-FINE-N: 2 levels, hidden 128, num_points [6,6], 3 layers.
TEST(DFINETransformerParity, MatchesDFineNDecoder) {
  DfTransformerConfig cfg;
  cfg.num_classes = 80;
  cfg.hidden_dim = 128;
  cfg.feat_channels = {128, 128};
  cfg.feat_strides = {16, 32};
  cfg.num_levels = 2;
  cfg.num_points = {6, 6};
  cfg.num_layers = 3;
  cfg.dim_feedforward = 512;
  CheckDecoder(cfg, 2, "/tmp/dfine_n_decoder.safetensors", "/tmp/dfine_n_dec/parity.safetensors");
}

// D-FINE-L: 3 levels, hidden 256, num_points [3,6,3], 6 layers.
TEST(DFINETransformerParity, MatchesDFineLDecoder) {
  DfTransformerConfig cfg;  // defaults are D-FINE-L
  CheckDecoder(cfg, 3, "/tmp/dfine_l_decoder.safetensors", "/tmp/dfine_l_dec/parity.safetensors");
}

// DEIMv2-Atto: 2 levels, hidden 64, num_points [4,2], 3 layers, RMSNorm + SwiGLU,
// no enc_output, no gateway (plain norm2).
TEST(DFINETransformerParity, MatchesDeimv2AttoDecoder) {
  DfTransformerConfig cfg;
  cfg.num_classes = 80;
  cfg.hidden_dim = 64;
  cfg.feat_channels = {64, 64};
  cfg.feat_strides = {16, 32};
  cfg.num_levels = 2;
  cfg.num_points = {4, 2};
  cfg.num_layers = 3;
  cfg.dim_feedforward = 160;
  cfg.silu = true;
  cfg.deimv2 = true;
  cfg.use_gateway = false;
  CheckDecoder(cfg, 2, "/tmp/deimv2_atto_decoder.safetensors", "/tmp/deimv2_atto_dec/parity.safetensors");
}

}  // namespace
}  // namespace detr::models
