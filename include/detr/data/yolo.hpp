// Copyright 2026 detrcpp authors. Apache-2.0.
//
// YOLO-format adapter. Reads a `data.yaml` (class names + per-split image
// locations) and the per-image `.txt` label files. YOLO labels are already in
// normalized center form (class cx cy w h), so boxes map directly to our BBox;
// image dimensions are left 0 (filled later when the image is decoded).

#pragma once

#include <filesystem>

#include "detr/core/result.hpp"
#include "detr/data/dataset.hpp"

namespace detr::data {

// Loads a YOLO dataset rooted at a directory containing data.yaml (or pointed
// directly at the data.yaml file).
core::Result<Dataset> LoadYolo(const std::filesystem::path& root);

}  // namespace detr::data
