// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/onnxexport/graph_builder.hpp"

#include <filesystem>
#include <vector>

#include <gtest/gtest.h>

namespace detr::onnxexport {
namespace {

TEST(OnnxBuilder, EmitsValidGemmReluGraph) {
  GraphBuilder b("tiny");
  b.AddInput("X", {1, 4});
  // Gemm: Y0 = X * W^T + B, with W [3,4], B [3].
  b.AddInitializerF32("W", {3, 4},
                      {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F, 0.9F, 1.0F, 1.1F, 1.2F});
  b.AddInitializerF32("B", {3}, {0.0F, 0.0F, 0.0F});
  b.AddNode("Gemm", {"X", "W", "B"}, {"g"}, "gemm0");
  b.AttrInt("transB", 1);
  b.AddNode("Relu", {"g"}, {"Y"}, "relu0");
  b.AddOutput("Y", {1, 3});

  const auto path = (std::filesystem::temp_directory_path() / "detr_tiny.onnx").string();
  auto r = b.Save(path, /*opset=*/17);
  ASSERT_TRUE(r.has_value()) << r.error().message;

  // File exists and is non-trivial.
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  EXPECT_FALSE(ec);
  EXPECT_GT(size, 32U);

  std::filesystem::remove(path, ec);
}

TEST(OnnxBuilder, UniqueNamesIncrement) {
  GraphBuilder b("g");
  EXPECT_EQ(b.Unique("conv"), "conv_0");
  EXPECT_EQ(b.Unique("conv"), "conv_1");
  EXPECT_EQ(b.Unique("bn"), "bn_0");
}

}  // namespace
}  // namespace detr::onnxexport
