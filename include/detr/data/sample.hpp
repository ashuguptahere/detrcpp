// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Core dataset value types, shared by every format adapter (COCO, YOLO, the
// native "detr" Parquet format). Boxes are stored in DETR's canonical form:
// normalized center-x, center-y, width, height in [0,1], class id 0-based. Every
// adapter converts its on-disk convention into this one, so the model and loss
// code never branch on format.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace detr::data {

enum class Split : std::uint8_t { Train, Val, Test };

std::string_view ToString(Split s);

// Normalized center form (cx, cy, w, h) in [0,1], matching DETR's target boxes.
struct BBox {
  float cx{0};
  float cy{0};
  float w{0};
  float h{0};
  int class_id{0};
};

struct Sample {
  std::string image_path;
  int width{0};
  int height{0};
  std::vector<BBox> boxes;
  Split split{Split::Train};
};

}  // namespace detr::data
