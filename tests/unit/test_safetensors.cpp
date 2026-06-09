// Copyright 2026 detrcpp authors. Apache-2.0.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "detr/weights/safetensors.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/tensor.hpp"

namespace detr::weights {
namespace {

std::vector<std::byte> BytesOf(const std::vector<float>& v) {
  std::vector<std::byte> b(v.size() * sizeof(float));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

RawTensor MakeF32(std::vector<std::int64_t> shape, std::vector<float> vals) {
  RawTensor t;
  t.dtype = DType::F32;
  t.shape = std::move(shape);
  t.data = BytesOf(vals);
  return t;
}

std::filesystem::path TempFile() {
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         ("detr_st_" + std::to_string(++counter) + ".safetensors");
}

TEST(Safetensors, RoundTripPreservesEverything) {
  StateDict sd;
  sd.Set("backbone.conv1.weight", MakeF32({2, 3}, {1, 2, 3, 4, 5, 6}));
  sd.Set("class_embed.bias", MakeF32({4}, {0.5F, -1.5F, 2.25F, 100.0F}));

  RawTensor idx;
  idx.dtype = DType::I64;
  idx.shape = {3};
  std::vector<std::int64_t> ivals{7, 8, 9};
  idx.data.resize(ivals.size() * sizeof(std::int64_t));
  std::memcpy(idx.data.data(), ivals.data(), idx.data.size());
  sd.Set("query_embed.weight", std::move(idx));

  sd.SetMeta("format", "pt");
  sd.SetMeta("detrcpp_version", "0.1.0");

  const auto path = TempFile();
  auto save = SaveSafetensors(path, sd);
  ASSERT_TRUE(save.has_value()) << save.error().message;

  auto loaded = LoadSafetensors(path);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  const StateDict& got = *loaded;

  EXPECT_EQ(got.Size(), 3U);
  // Insertion order preserved.
  ASSERT_EQ(got.Keys().size(), 3U);
  EXPECT_EQ(got.Keys()[0], "backbone.conv1.weight");
  EXPECT_EQ(got.Keys()[1], "class_embed.bias");
  EXPECT_EQ(got.Keys()[2], "query_embed.weight");

  const RawTensor* w = got.Find("backbone.conv1.weight");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->dtype, DType::F32);
  EXPECT_EQ(w->shape, (std::vector<std::int64_t>{2, 3}));
  std::vector<float> back(6);
  std::memcpy(back.data(), w->data.data(), w->data.size());
  EXPECT_EQ(back, (std::vector<float>{1, 2, 3, 4, 5, 6}));

  const RawTensor* q = got.Find("query_embed.weight");
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->dtype, DType::I64);
  std::vector<std::int64_t> qback(3);
  std::memcpy(qback.data(), q->data.data(), q->data.size());
  EXPECT_EQ(qback, (std::vector<std::int64_t>{7, 8, 9}));

  EXPECT_EQ(got.GetMeta("format").value_or(""), "pt");
  EXPECT_EQ(got.GetMeta("detrcpp_version").value_or(""), "0.1.0");

  std::filesystem::remove(path);
}

TEST(Safetensors, RoundTripsScalarTensor) {
  // A rank-0 scalar (empty shape, numel 1) — e.g. BatchNorm's
  // num_batches_tracked. Regression guard: empty shape must mean numel 1.
  StateDict sd;
  RawTensor s;
  s.dtype = DType::I64;
  s.shape = {};  // scalar
  std::int64_t v = 123;
  s.data.resize(sizeof(std::int64_t));
  std::memcpy(s.data.data(), &v, sizeof(v));
  EXPECT_EQ(s.Numel(), 1);
  EXPECT_EQ(s.Nbytes(), sizeof(std::int64_t));
  sd.Set("bn.num_batches_tracked", std::move(s));

  const auto path = TempFile();
  ASSERT_TRUE(SaveSafetensors(path, sd).has_value());
  auto loaded = LoadSafetensors(path);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  const RawTensor* got = loaded->Find("bn.num_batches_tracked");
  ASSERT_NE(got, nullptr);
  EXPECT_TRUE(got->shape.empty());
  std::int64_t back = 0;
  std::memcpy(&back, got->data.data(), sizeof(back));
  EXPECT_EQ(back, 123);
  std::filesystem::remove(path);
}

TEST(Safetensors, RejectsTruncatedFile) {
  const auto path = TempFile();
  {
    std::ofstream f(path, std::ios::binary);
    const char tiny[4] = {1, 0, 0, 0};
    f.write(tiny, 4);
  }
  auto loaded = LoadSafetensors(path);
  EXPECT_FALSE(loaded.has_value());
  std::filesystem::remove(path);
}

TEST(Safetensors, SaveRejectsByteLengthMismatch) {
  StateDict sd;
  RawTensor bad;
  bad.dtype = DType::F32;
  bad.shape = {10};    // claims 40 bytes
  bad.data.resize(8);  // but only 8
  sd.Set("bad", std::move(bad));
  auto save = SaveSafetensors(TempFile(), sd);
  EXPECT_FALSE(save.has_value());
}

}  // namespace
}  // namespace detr::weights
