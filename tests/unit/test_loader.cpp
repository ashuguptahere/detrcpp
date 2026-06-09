// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Exercises the real image path end to end: write tiny PPM images, decode +
// resize + normalize + collate them with DataLoader, and confirm the resulting
// batch drives a Trainer step. PPM (P6) is used because stb_image decodes it, so
// no encoder is needed in the test.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "detr/data/dataset.hpp"
#include "detr/data/loader.hpp"
#include "detr/data/sample.hpp"
#include "detr/models/detr.hpp"
#include "detr/models/registry.hpp"
#include "detr/train/trainer.hpp"

namespace detr::data {
namespace {

std::string WritePpm(const std::filesystem::path& p, int w, int h, unsigned char r, unsigned char g,
                     unsigned char b) {
  std::ofstream f(p, std::ios::binary);
  f << "P6\n" << w << " " << h << "\n255\n";
  std::vector<unsigned char> px(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3);
  for (std::size_t i = 0; i < px.size(); i += 3) {
    px[i] = r;
    px[i + 1] = g;
    px[i + 2] = b;
  }
  f.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
  return p.string();
}

Dataset MakeTinyDataset(const std::filesystem::path& dir) {
  std::filesystem::create_directories(dir);
  Dataset ds;
  ds.class_names = {"a", "b", "c", "d"};
  for (int i = 0; i < 3; ++i) {
    Sample s;
    s.image_path = WritePpm(dir / ("img" + std::to_string(i) + ".ppm"), 8 + i, 10 + i,
                            static_cast<unsigned char>(i * 40), 100, 200);
    s.split = Split::Train;
    s.boxes = {BBox{0.5F, 0.5F, 0.4F, 0.4F, i % 4}};
    ds.samples.push_back(s);
  }
  return ds;
}

TEST(DataLoader, DecodesResizesAndBatches) {
  const auto dir = std::filesystem::temp_directory_path() / "detr_loader_test";
  auto ds = MakeTinyDataset(dir);

  DataLoader loader(ds, Split::Train, /*imgsz=*/16, /*batch=*/2, /*seed=*/42);
  EXPECT_EQ(loader.NumSamples(), 3U);
  EXPECT_EQ(loader.NumBatches(), 2U);  // ceil(3/2)

  auto b0 = loader.At(0);
  ASSERT_TRUE(b0.has_value()) << b0.error().message;
  EXPECT_EQ(b0->images.sizes(), (std::vector<std::int64_t>{2, 3, 16, 16}));
  EXPECT_EQ(b0->targets.size(), 2U);
  EXPECT_EQ(b0->targets[0].boxes.size(0), 1);
  EXPECT_GT(b0->images.abs().sum().item<float>(), 0.0F);  // not blank

  auto b1 = loader.At(1);
  ASSERT_TRUE(b1.has_value());
  EXPECT_EQ(b1->images.size(0), 1);  // last partial batch

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(DataLoader, FeedsARealTrainerStep) {
  models::RegisterBuiltins();
  const auto dir = std::filesystem::temp_directory_path() / "detr_loader_train";
  auto ds = MakeTinyDataset(dir);

  YAML::Node cfg;
  cfg["hidden_dim"] = 32;
  cfg["nheads"] = 4;
  cfg["enc_layers"] = 1;
  cfg["dec_layers"] = 1;
  cfg["dim_feedforward"] = 64;
  cfg["num_queries"] = 8;
  cfg["num_classes"] = 4;
  cfg["backbone_width"] = 8;
  cfg["imgsz"] = 16;
  auto model = models::Registry::Instance().Build("detr", cfg);
  ASSERT_TRUE(model.has_value()) << model.error().message;

  train::TrainConfig tc;
  tc.lr = 1e-3;
  train::Trainer trainer(*model, tc);

  DataLoader loader(ds, Split::Train, 16, 2, 7);
  auto batch = loader.At(0);
  ASSERT_TRUE(batch.has_value()) << batch.error().message;
  const float loss = trainer.TrainStep(batch->images, batch->targets);
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_GE(loss, 0.0F);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

}  // namespace
}  // namespace detr::data
