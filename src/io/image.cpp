// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/io/image.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>

#include "stb_image.h"
#include "stb_image_write.h"

namespace detr::io {

using core::Err;
using core::ErrorCode;
using core::Result;

Result<RgbImage> LoadRgb(const std::filesystem::path& path) {
  int w = 0;
  int h = 0;
  int comp = 0;
  unsigned char* pixels = stbi_load(path.string().c_str(), &w, &h, &comp, 3);
  if (pixels == nullptr || w <= 0 || h <= 0) {
    if (pixels != nullptr) {
      stbi_image_free(pixels);
    }
    return Err(ErrorCode::Io, fmt::format("cannot read image '{}'", path.string()));
  }
  RgbImage img;
  img.width = w;
  img.height = h;
  const auto n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
  img.data.assign(pixels, pixels + n);
  stbi_image_free(pixels);
  return img;
}

Result<void> SavePng(const std::filesystem::path& path, const RgbImage& img) {
  if (img.width <= 0 || img.height <= 0) {
    return Err(ErrorCode::InvalidArgument, "empty image");
  }
  const int ok = stbi_write_png(path.string().c_str(), img.width, img.height, 3, img.data.data(),
                                img.width * 3);
  if (ok == 0) {
    return Err(ErrorCode::Io, fmt::format("cannot write png '{}'", path.string()));
  }
  return {};
}

void DrawRect(RgbImage& img, int x0, int y0, int x1, int y1, std::uint8_t r, std::uint8_t g,
              std::uint8_t b, int thickness) {
  if (img.width <= 0 || img.height <= 0) {
    return;
  }
  x0 = std::clamp(x0, 0, img.width - 1);
  x1 = std::clamp(x1, 0, img.width - 1);
  y0 = std::clamp(y0, 0, img.height - 1);
  y1 = std::clamp(y1, 0, img.height - 1);
  if (x1 < x0) {
    std::swap(x0, x1);
  }
  if (y1 < y0) {
    std::swap(y0, y1);
  }
  auto put = [&](int x, int y) {
    if (x < 0 || y < 0 || x >= img.width || y >= img.height) {
      return;
    }
    const auto idx = (static_cast<std::size_t>(y) * static_cast<std::size_t>(img.width) +
                      static_cast<std::size_t>(x)) *
                     3;
    img.data[idx] = r;
    img.data[idx + 1] = g;
    img.data[idx + 2] = b;
  };
  for (int t = 0; t < thickness; ++t) {
    for (int x = x0; x <= x1; ++x) {
      put(x, y0 + t);
      put(x, y1 - t);
    }
    for (int y = y0; y <= y1; ++y) {
      put(x0 + t, y);
      put(x1 - t, y);
    }
  }
}

}  // namespace detr::io
