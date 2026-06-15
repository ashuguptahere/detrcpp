// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Round-trips the torch-free .pth reader/writer and checks byte-level interop with
// PyTorch: the file written here is also loaded by `torch.load` in a /tmp helper
// (see the InteropFixture test, which writes a stable path).

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "detr/weights/pth.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/tensor.hpp"

namespace detr::weights {
namespace {

RawTensor MakeF32(std::vector<std::int64_t> shape, std::vector<float> vals) {
  RawTensor t;
  t.dtype = DType::F32;
  t.shape = std::move(shape);
  t.data.resize(vals.size() * sizeof(float));
  std::memcpy(t.data.data(), vals.data(), t.data.size());
  return t;
}

RawTensor MakeI64(std::vector<std::int64_t> shape, std::vector<std::int64_t> vals) {
  RawTensor t;
  t.dtype = DType::I64;
  t.shape = std::move(shape);
  t.data.resize(vals.size() * sizeof(std::int64_t));
  std::memcpy(t.data.data(), vals.data(), t.data.size());
  return t;
}

std::filesystem::path TempFile() {
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         ("detr_pth_" + std::to_string(static_cast<long>(::getpid())) + "_" +
          std::to_string(++counter) + ".pth");
}

TEST(Pth, RoundTripPreservesTensors) {
  StateDict sd;
  sd.Set("backbone.conv1.weight", MakeF32({2, 3}, {1, 2, 3, 4, 5, 6}));
  sd.Set("class_embed.bias", MakeF32({4}, {0.5F, -1.5F, 2.25F, 100.0F}));
  sd.Set("scalar", MakeF32({}, {42.0F}));  // rank-0
  sd.Set("query_embed.weight", MakeI64({3}, {7, 8, 9}));

  const auto path = TempFile();
  auto save = SavePth(path, sd);
  ASSERT_TRUE(save.has_value()) << save.error().message;

  auto loaded = LoadPth(path);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  const StateDict& got = *loaded;

  ASSERT_EQ(got.Keys().size(), 4U);
  // Insertion order preserved.
  EXPECT_EQ(got.Keys()[0], "backbone.conv1.weight");
  EXPECT_EQ(got.Keys()[3], "query_embed.weight");
  for (const std::string& k : sd.Keys()) {
    const RawTensor* a = sd.Find(k);
    const RawTensor* b = got.Find(k);
    ASSERT_NE(b, nullptr) << k;
    EXPECT_EQ(a->dtype, b->dtype) << k;
    EXPECT_EQ(a->shape, b->shape) << k;
    ASSERT_EQ(a->data.size(), b->data.size()) << k;
    EXPECT_EQ(std::memcmp(a->data.data(), b->data.data(), a->data.size()), 0) << k;
  }
  std::filesystem::remove(path);
}

// Writes a stable .pth that a /tmp Python helper torch.loads to confirm interop.
TEST(Pth, InteropFixtureForTorchLoad) {
  StateDict sd;
  sd.Set("w", MakeF32({2, 2}, {1.5F, 2.5F, 3.5F, 4.5F}));
  sd.Set("b", MakeI64({3}, {10, 20, 30}));
  const auto path = std::filesystem::temp_directory_path() / "detr_pth_interop.pth";
  auto save = SavePth(path, sd);
  ASSERT_TRUE(save.has_value()) << save.error().message;
  // Read our own output back as a sanity check; Python torch.load is checked externally.
  auto loaded = LoadPth(path);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  EXPECT_EQ(loaded->Keys().size(), 2U);
}

}  // namespace
}  // namespace detr::weights
