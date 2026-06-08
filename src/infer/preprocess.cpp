// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/infer/preprocess.hpp"

#include <cstddef>
#include <vector>

#include "stb_image_resize2.h"

namespace detr::infer {

torch::Tensor PreprocessImage(const io::RgbImage& img, int imgsz) {
  std::vector<unsigned char> resized(static_cast<std::size_t>(imgsz) *
                                     static_cast<std::size_t>(imgsz) * 3);
  stbir_resize_uint8_linear(img.data.data(), img.width, img.height, 0, resized.data(), imgsz,
                            imgsz, 0, STBIR_RGB);
  auto hwc = torch::from_blob(resized.data(), {imgsz, imgsz, 3}, torch::kUInt8).clone();
  auto chw = hwc.to(torch::kFloat32).div_(255.0).permute({2, 0, 1}).contiguous();
  static const auto mean = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
  static const auto stddev = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
  return ((chw - mean) / stddev).unsqueeze(0);  // [1, 3, imgsz, imgsz]
}

}  // namespace detr::infer
