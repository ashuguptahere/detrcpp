// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DataLoader: turns a Dataset split into training batches. It decodes each image
// (stb_image), resizes to a square imgsz, normalizes with ImageNet statistics,
// and stacks into [B,3,H,W]; targets come straight from the Sample's normalized
// cxcywh boxes (square resize is per-axis, so normalized boxes are unchanged).
// Batch order is seed-reproducible. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <vector>

#include "detr/core/result.hpp"
#include "detr/data/dataset.hpp"
#include "detr/data/sample.hpp"
#include "detr/train/target.hpp"

namespace detr::data {

struct Batch {
  torch::Tensor images;                     // [B, 3, imgsz, imgsz]
  train::TargetBatch targets;               // B entries
  std::vector<std::pair<int, int>> sizes;   // original (width, height) per image
  std::vector<std::size_t> sample_indices;  // index into Dataset::samples
};

class DataLoader {
 public:
  DataLoader(Dataset dataset, Split split, int imgsz, int batch_size, std::uint64_t seed);

  std::size_t NumBatches() const;
  std::size_t NumSamples() const { return indices_.size(); }
  int ImgSize() const { return imgsz_; }

  // Reorders the split's sample indices for a new epoch (seed-reproducible).
  void Reshuffle(std::uint64_t epoch_seed);

  // Decodes + collates batch |i| (0 <= i < NumBatches()). Unreadable images
  // become zero tensors with empty targets, with a logged warning.
  core::Result<Batch> At(std::size_t i) const;

 private:
  Dataset dataset_;
  std::vector<std::size_t> indices_;  // sample indices belonging to the split
  int imgsz_;
  int batch_;
};

}  // namespace detr::data
