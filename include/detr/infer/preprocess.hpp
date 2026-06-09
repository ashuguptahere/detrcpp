// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Turns an RgbImage into a model input tensor: resize to a square imgsz,
// scale to [0,1], ImageNet-normalize, and add a batch dim -> [1,3,imgsz,imgsz].
// Matches the training-time preprocessing. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include "detr/io/image.hpp"

namespace detr::infer {

// Square resize to imgsz x imgsz (training-time preprocessing).
torch::Tensor PreprocessImage(const io::RgbImage& img, int imgsz);

// DETR's eval preprocessing: resize so the shortest side is |short_side|,
// preserving aspect ratio, but never letting the longest side exceed |max_size|.
// Returns [1, 3, H, W] (variable H, W). Use at batch 1 (no padding/mask needed).
torch::Tensor PreprocessImageAspect(const io::RgbImage& img, int short_side, int max_size = 1333);

}  // namespace detr::infer
