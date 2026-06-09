// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/infer/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "stb_image_resize2.h"

namespace detr::infer {

namespace {

// Resized uint8 HWC -> normalized [1,3,H,W] float tensor.
torch::Tensor ToInput(const std::vector<unsigned char>& rgb, int h, int w) {
  auto hwc = torch::from_blob(const_cast<unsigned char*>(rgb.data()), {h, w, 3}, torch::kUInt8)
                 .clone();
  auto chw = hwc.to(torch::kFloat32).div_(255.0).permute({2, 0, 1}).contiguous();
  static const auto mean = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
  static const auto stddev = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
  return ((chw - mean) / stddev).unsqueeze(0);
}

}  // namespace

torch::Tensor PreprocessImage(const io::RgbImage& img, int imgsz) {
  std::vector<unsigned char> resized(static_cast<std::size_t>(imgsz) *
                                     static_cast<std::size_t>(imgsz) * 3);
  stbir_resize_uint8_linear(img.data.data(), img.width, img.height, 0, resized.data(), imgsz,
                            imgsz, 0, STBIR_RGB);
  return ToInput(resized, imgsz, imgsz);  // [1, 3, imgsz, imgsz]
}

torch::Tensor PreprocessImageAspect(const io::RgbImage& img, int short_side, int max_size) {
  const int w = img.width;
  const int h = img.height;
  double scale = static_cast<double>(short_side) / std::min(w, h);
  if (std::max(w, h) * scale > max_size) {
    scale = static_cast<double>(max_size) / std::max(w, h);
  }
  const int nw = std::max(1, static_cast<int>(std::lround(w * scale)));
  const int nh = std::max(1, static_cast<int>(std::lround(h * scale)));
  std::vector<unsigned char> resized(static_cast<std::size_t>(nw) *
                                     static_cast<std::size_t>(nh) * 3);
  stbir_resize_uint8_linear(img.data.data(), w, h, 0, resized.data(), nw, nh, 0, STBIR_RGB);
  return ToInput(resized, nh, nw);  // [1, 3, nh, nw]
}

}  // namespace detr::infer
