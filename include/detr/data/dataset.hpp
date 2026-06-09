// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Dataset: the format-independent in-memory view a Trainer/Evaluator consumes —
// a flat list of Samples plus the class-name table. Every adapter (COCO, YOLO,
// native "detr" Parquet) produces one of these, so nothing downstream branches
// on the source format.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "detr/core/result.hpp"
#include "detr/data/sample.hpp"

namespace detr::data {

enum class Format : std::uint8_t { Auto, Coco, Yolo, Detr };

std::string_view ToString(Format f);

struct Dataset {
  std::vector<Sample> samples;
  std::vector<std::string> class_names;

  std::size_t Size() const { return samples.size(); }
  bool Empty() const { return samples.empty(); }

  // Number of samples in a given split.
  std::size_t CountOf(Split s) const;

  // Indices of samples belonging to |s| (in current order).
  std::vector<std::size_t> IndicesOf(Split s) const;

  // Deterministic, seed-reproducible shuffle. The result depends only on |seed|
  // and the set of samples (not their incoming order): samples are first sorted
  // by image_path into a canonical order, then permuted with mt19937_64(seed).
  // Two runs with the same seed produce the identical ordering.
  void Shuffle(std::uint64_t seed);
};

// Detects the dataset format at |root| (COCO annotations json, YOLO data.yaml,
// or a .detr/.parquet file). Returns Format::Auto if undetermined.
Format DetectFormat(const std::filesystem::path& root);

// Loads a dataset from |root|, dispatching on |format| (Auto -> DetectFormat).
// |raw_coco_ids| keeps raw COCO category ids (for evaluating COCO-91 models like
// official DETR); ignored for non-COCO formats.
core::Result<Dataset> LoadDataset(const std::filesystem::path& root, Format format = Format::Auto,
                                  bool raw_coco_ids = false);

}  // namespace detr::data
