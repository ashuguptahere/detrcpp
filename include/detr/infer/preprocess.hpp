// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Turns an RgbImage into a model input tensor: resize to a square imgsz,
// scale to [0,1], ImageNet-normalize, and add a batch dim -> [1,3,imgsz,imgsz].
// Matches the training-time preprocessing. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include "detr/io/image.hpp"

namespace detr::infer {

torch::Tensor PreprocessImage(const io::RgbImage& img, int imgsz);

}  // namespace detr::infer
