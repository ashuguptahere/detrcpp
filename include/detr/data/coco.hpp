// Copyright 2026 detrcpp authors. Apache-2.0.
//
// COCO-format adapter. Parses an `instances_*.json` annotation file with
// simdjson (SIMD-accelerated) into our Dataset. COCO stores boxes as absolute
// [x, y, w, h] with a top-left origin and non-contiguous category ids; we
// convert to normalized center form and remap category ids to a 0-based,
// id-sorted class index.

#pragma once

#include <filesystem>

#include "detr/core/result.hpp"
#include "detr/data/dataset.hpp"
#include "detr/data/sample.hpp"

namespace detr::data {

// Parses one COCO annotation json. |images_dir| is prepended to each
// image file_name to form Sample::image_path. All produced samples are tagged
// with |split|.
core::Result<Dataset> LoadCocoJson(const std::filesystem::path& annotations_json,
                                   const std::filesystem::path& images_dir, Split split);

}  // namespace detr::data
