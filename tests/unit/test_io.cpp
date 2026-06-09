// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Image I/O + source resolution + the full predict primitives (preprocess ->
// model -> postprocess -> draw -> save).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "detr/infer/postprocess.hpp"
#include "detr/infer/preprocess.hpp"
#include "detr/io/image.hpp"
#include "detr/io/source.hpp"
#include "detr/models/registry.hpp"

namespace detr {
namespace {

std::string WritePpm(const std::filesystem::path& p, int w, int h) {
  std::ofstream f(p, std::ios::binary);
  f << "P6\n" << w << " " << h << "\n255\n";
  std::vector<unsigned char> px(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3, 120);
  f.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
  return p.string();
}

TEST(Image, LoadDrawSaveRoundTrip) {
  const auto dir = std::filesystem::temp_directory_path() / "detr_io_test";
  std::filesystem::create_directories(dir);
  const auto src = WritePpm(dir / "a.ppm", 20, 16);

  auto img = io::LoadRgb(src);
  ASSERT_TRUE(img.has_value()) << img.error().message;
  EXPECT_EQ(img->width, 20);
  EXPECT_EQ(img->height, 16);
  EXPECT_EQ(img->data.size(), 20U * 16U * 3U);

  io::DrawRect(*img, 2, 2, 10, 10, 255, 0, 0, 2);
  const auto out = dir / "a_out.png";
  ASSERT_TRUE(io::SavePng(out, *img).has_value());

  auto reloaded = io::LoadRgb(out);
  ASSERT_TRUE(reloaded.has_value());
  EXPECT_EQ(reloaded->width, 20);
  EXPECT_EQ(reloaded->height, 16);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Source, ResolvesFileDirGlobAndRejectsUrl) {
  const auto dir = std::filesystem::temp_directory_path() / "detr_src_test";
  std::filesystem::create_directories(dir);
  WritePpm(dir / "one.ppm", 8, 8);
  WritePpm(dir / "two.ppm", 8, 8);

  auto single = io::ResolveImageSources((dir / "one.ppm").string());
  ASSERT_TRUE(single.has_value());
  EXPECT_EQ(single->size(), 1U);

  auto whole_dir = io::ResolveImageSources(dir.string());
  ASSERT_TRUE(whole_dir.has_value());
  EXPECT_EQ(whole_dir->size(), 2U);

  auto glob = io::ResolveImageSources((dir / "*.ppm").string());
  ASSERT_TRUE(glob.has_value());
  EXPECT_EQ(glob->size(), 2U);

  EXPECT_FALSE(io::ResolveImageSources("https://example.com/x.jpg").has_value());

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Predict, PreprocessForwardPostprocessRuns) {
  models::RegisterBuiltins();
  YAML::Node cfg;
  cfg["hidden_dim"] = 32;
  cfg["nheads"] = 4;
  cfg["enc_layers"] = 1;
  cfg["dec_layers"] = 1;
  cfg["dim_feedforward"] = 64;
  cfg["num_queries"] = 6;
  cfg["num_classes"] = 4;
  cfg["imgsz"] = 32;
  cfg["backbone_width"] = 8;
  auto model = models::Registry::Instance().Build("detr", cfg);
  ASSERT_TRUE(model.has_value()) << model.error().message;
  (*model)->eval();
  torch::NoGradGuard ng;

  io::RgbImage img;
  img.width = 24;
  img.height = 18;
  img.data.assign(static_cast<std::size_t>(24) * 18 * 3, 100);

  auto input = infer::PreprocessImage(img, 32);
  EXPECT_EQ(input.sizes(), (std::vector<std::int64_t>{1, 3, 32, 32}));
  auto out = (*model)->Forward(input);
  auto dets = infer::PostprocessImage(out, 0, img.width, img.height, 4);
  // One detection per query, each with a class in range and a box in image bounds-ish.
  EXPECT_EQ(dets.size(), 6U);
  for (const auto& d : dets) {
    EXPECT_GE(d.category_id, 0);
    EXPECT_LT(d.category_id, 4);
    EXPECT_GE(d.score, 0.0F);
    EXPECT_LE(d.score, 1.0F);
  }
}

}  // namespace
}  // namespace detr
