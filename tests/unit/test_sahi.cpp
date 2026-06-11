// Copyright 2026 detrcpp authors. Apache-2.0.
//
// SAHI: overlapping-tile slicing math, the new class-aware NMS, and the end-to-end
// shift-back + cross-tile merge (a planted object seen by two overlapping tiles
// collapses to a single full-image box). No real model — a stub tile detector.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

#include "detr/infer/sahi.hpp"

namespace detr::infer {
namespace {

TEST(Sahi, TilingCountAndCoverage) {
  SahiConfig cfg;  // slice 512, overlap 0.2 -> step 410
  auto s = ComputeSlices(1000, 800, cfg);
  EXPECT_EQ(s.size(), 6U);  // 3 cols x 2 rows
  for (const auto& t : s) {
    EXPECT_EQ(t.x1 - t.x0, 512);
    EXPECT_EQ(t.y1 - t.y0, 512);
  }
  EXPECT_EQ(s.front().x0, 0);
  EXPECT_EQ(s.front().y0, 0);
  EXPECT_EQ(s.back().x1, 1000);  // last tile flush to the border
  EXPECT_EQ(s.back().y1, 800);

  // Smaller than a slice on both axes -> a single tile spanning the image.
  auto one = ComputeSlices(200, 200, cfg);
  ASSERT_EQ(one.size(), 1U);
  EXPECT_EQ(one[0].x0, 0);
  EXPECT_EQ(one[0].x1, 200);
  EXPECT_EQ(one[0].y1, 200);
}

TEST(Sahi, NmsPerClassIsClassAware) {
  // Two near-identical boxes, same class -> one survives.
  std::vector<eval::DtBox> same{{0, 10.0F, 10.0F, 40.0F, 40.0F, 0.9F},
                                {0, 12.0F, 11.0F, 40.0F, 40.0F, 0.8F}};
  EXPECT_EQ(NmsPerClass(same, 0.5F).size(), 1U);

  // Same overlap but different classes -> both survive.
  std::vector<eval::DtBox> diff{{0, 10.0F, 10.0F, 40.0F, 40.0F, 0.9F},
                                {1, 12.0F, 11.0F, 40.0F, 40.0F, 0.8F}};
  EXPECT_EQ(NmsPerClass(diff, 0.5F).size(), 2U);
}

// A stub tile detector: returns the bounding box (crop-local xywh) of the bright
// (non-zero) pixels in the crop, as one class-0 detection.
std::vector<eval::DtBox> BrightBlobDetector(const io::RgbImage& tile) {
  int x0 = tile.width;
  int y0 = tile.height;
  int x1 = -1;
  int y1 = -1;
  for (int y = 0; y < tile.height; ++y) {
    for (int x = 0; x < tile.width; ++x) {
      if (tile.data[(static_cast<std::size_t>(y) * static_cast<std::size_t>(tile.width) +
                     static_cast<std::size_t>(x)) * 3] > 0) {
        x0 = std::min(x0, x);
        y0 = std::min(y0, y);
        x1 = std::max(x1, x + 1);
        y1 = std::max(y1, y + 1);
      }
    }
  }
  if (x1 < 0) {
    return {};
  }
  return {eval::DtBox{0, static_cast<float>(x0), static_cast<float>(y0),
                      static_cast<float>(x1 - x0), static_cast<float>(y1 - y0), 0.9F}};
}

TEST(Sahi, ShiftBackAndMergeAcrossTiles) {
  // 1000x800 image, a 20x20 bright block planted in the x-overlap region so two
  // neighboring tiles (cols 0 and 1) both see it.
  io::RgbImage img;
  img.width = 1000;
  img.height = 800;
  img.data.assign(static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height) * 3, 0);
  const int px = 440;
  const int py = 100;
  for (int y = py; y < py + 20; ++y) {
    for (int x = px; x < px + 20; ++x) {
      img.data[(static_cast<std::size_t>(y) * static_cast<std::size_t>(img.width) +
                static_cast<std::size_t>(x)) * 3] = 255;
    }
  }

  SahiConfig cfg;
  auto dets = SahiDetect(img, BrightBlobDetector, cfg);
  ASSERT_EQ(dets.size(), 1U);  // the two tile detections merged into one
  EXPECT_NEAR(dets[0].x, static_cast<float>(px), 1.0F);
  EXPECT_NEAR(dets[0].y, static_cast<float>(py), 1.0F);
  EXPECT_NEAR(dets[0].w, 20.0F, 1.0F);
  EXPECT_NEAR(dets[0].h, 20.0F, 1.0F);
}

}  // namespace
}  // namespace detr::infer
