// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Proves the weight-interop round trip on a real LibTorch module: extract a
// module's weights -> safetensors -> load back into a freshly-initialized
// module by name, and confirm the parameters are identical. Also proves the
// WeightRemapper adapts upstream parameter names (e.g. a "model." prefix).

#include "detr/weights/torch_bridge.hpp"

#include <filesystem>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "detr/weights/remapper.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/state_dict.hpp"

namespace detr::weights {
namespace {

struct TinyNet : torch::nn::Module {
  TinyNet() {
    fc1 = register_module("fc1", torch::nn::Linear(4, 3));
    fc2 = register_module("fc2", torch::nn::Linear(3, 2));
  }
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
};

TEST(TorchBridge, RoundTripThroughSafetensors) {
  torch::manual_seed(0);
  TinyNet a;

  StateDict sd = StateDictFromModule(a);
  // fc1.weight, fc1.bias, fc2.weight, fc2.bias
  ASSERT_EQ(sd.Size(), 4U);

  const auto path =
      std::filesystem::temp_directory_path() / "detr_bridge_roundtrip.safetensors";
  ASSERT_TRUE(SaveSafetensors(path, sd).has_value());
  auto loaded = LoadSafetensors(path);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;

  torch::manual_seed(1);
  TinyNet b;  // different random init
  ASSERT_FALSE(torch::allclose(a.fc1->weight, b.fc1->weight));

  auto rep = LoadStateDictInto(b, *loaded, WeightRemapper{}, /*strict=*/true);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  EXPECT_EQ(rep->loaded, 4U);
  EXPECT_TRUE(rep->missing.empty());
  EXPECT_TRUE(rep->unexpected.empty());
  EXPECT_TRUE(rep->mismatched.empty());

  EXPECT_TRUE(torch::allclose(a.fc1->weight, b.fc1->weight));
  EXPECT_TRUE(torch::allclose(a.fc1->bias, b.fc1->bias));
  EXPECT_TRUE(torch::allclose(a.fc2->weight, b.fc2->weight));
  EXPECT_TRUE(torch::allclose(a.fc2->bias, b.fc2->bias));

  std::filesystem::remove(path);
}

TEST(TorchBridge, RemapperAdaptsUpstreamNames) {
  torch::manual_seed(2);
  TinyNet a;
  StateDict sd = StateDictFromModule(a);

  // Simulate an upstream checkpoint that prefixes every key with "model.".
  StateDict upstream;
  for (const auto& k : sd.Keys()) {
    upstream.Set("model." + k, *sd.Find(k));
  }

  torch::manual_seed(3);
  TinyNet b;
  WeightRemapper remap;
  remap.StripPrefix("model.");
  auto rep = LoadStateDictInto(b, upstream, remap, /*strict=*/true);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;
  EXPECT_EQ(rep->loaded, 4U);
  EXPECT_TRUE(torch::allclose(a.fc1->weight, b.fc1->weight));
  EXPECT_TRUE(torch::allclose(a.fc2->bias, b.fc2->bias));
}

TEST(TorchBridge, DtypeRoundTrips) {
  auto t = torch::tensor({{1.5F, -2.0F}, {3.25F, 4.0F}}, torch::kFloat32);
  auto raw = FromTensor(t);
  ASSERT_TRUE(raw.has_value()) << raw.error().message;
  EXPECT_EQ(raw->dtype, DType::F32);
  EXPECT_EQ(raw->shape, (std::vector<std::int64_t>{2, 2}));
  auto back = ToTensor(*raw);
  EXPECT_TRUE(torch::allclose(t, back));
}

}  // namespace
}  // namespace detr::weights
