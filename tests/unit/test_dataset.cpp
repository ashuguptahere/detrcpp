// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/data/dataset.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "detr/data/coco.hpp"
#include "detr/data/sample.hpp"

namespace detr::data {
namespace {

std::filesystem::path WriteTemp(const std::string& name, const std::string& content) {
  auto p = std::filesystem::temp_directory_path() / name;
  std::ofstream f(p);
  f << content;
  return p;
}

// Minimal but valid COCO: 1 image 100x200, two categories (non-contiguous ids),
// two annotations (one crowd, which must be dropped).
constexpr const char* kCocoJson = R"({
  "images": [ {"id": 7, "file_name": "a.jpg", "width": 100, "height": 200} ],
  "categories": [ {"id": 3, "name": "cat"}, {"id": 1, "name": "dog"} ],
  "annotations": [
    {"image_id": 7, "category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
    {"image_id": 7, "category_id": 3, "bbox": [0, 0, 50, 50], "iscrowd": 1}
  ]
})";

TEST(Coco, ParsesAndNormalizes) {
  auto path = WriteTemp("detr_coco_test.json", kCocoJson);
  auto ds = LoadCocoJson(path, "/imgs", Split::Train);
  ASSERT_TRUE(ds.has_value()) << ds.error().message;

  // categories sorted by id -> dog(id1)=class0, cat(id3)=class1.
  ASSERT_EQ(ds->class_names.size(), 2U);
  EXPECT_EQ(ds->class_names[0], "dog");
  EXPECT_EQ(ds->class_names[1], "cat");

  ASSERT_EQ(ds->samples.size(), 1U);
  const Sample& s = ds->samples[0];
  EXPECT_EQ(s.image_path, std::filesystem::path("/imgs/a.jpg").string());
  EXPECT_EQ(s.width, 100);
  EXPECT_EQ(s.height, 200);

  // Crowd annotation dropped -> only one box.
  ASSERT_EQ(s.boxes.size(), 1U);
  const BBox& b = s.boxes[0];
  EXPECT_EQ(b.class_id, 0);                 // category_id 1 -> dog -> class 0
  EXPECT_FLOAT_EQ(b.cx, (10.0F + 15.0F) / 100.0F);  // (x + w/2)/W
  EXPECT_FLOAT_EQ(b.cy, (20.0F + 20.0F) / 200.0F);  // (y + h/2)/H
  EXPECT_FLOAT_EQ(b.w, 30.0F / 100.0F);
  EXPECT_FLOAT_EQ(b.h, 40.0F / 200.0F);

  std::filesystem::remove(path);
}

TEST(Dataset, ShuffleIsSeedReproducibleAndOrderIndependent) {
  Dataset a;
  Dataset b;
  for (int i = 0; i < 50; ++i) {
    Sample s;
    s.image_path = "img_" + std::to_string(i) + ".jpg";
    a.samples.push_back(s);
  }
  b.samples.assign(a.samples.rbegin(), a.samples.rend());  // reversed input order

  a.Shuffle(42);
  b.Shuffle(42);
  ASSERT_EQ(a.samples.size(), b.samples.size());
  for (std::size_t i = 0; i < a.samples.size(); ++i) {
    EXPECT_EQ(a.samples[i].image_path, b.samples[i].image_path)
        << "shuffle must depend only on seed + set, not input order, at " << i;
  }

  // Different seed -> (very likely) different order.
  Dataset c = a;
  c.Shuffle(43);
  bool differs = false;
  for (std::size_t i = 0; i < a.samples.size(); ++i) {
    if (a.samples[i].image_path != c.samples[i].image_path) {
      differs = true;
      break;
    }
  }
  EXPECT_TRUE(differs);
}

TEST(Dataset, SplitCounts) {
  Dataset d;
  Sample tr;
  tr.split = Split::Train;
  Sample va;
  va.split = Split::Val;
  d.samples = {tr, tr, va};
  EXPECT_EQ(d.CountOf(Split::Train), 2U);
  EXPECT_EQ(d.CountOf(Split::Val), 1U);
  EXPECT_EQ(d.IndicesOf(Split::Val).size(), 1U);
}

}  // namespace
}  // namespace detr::data
