// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Minimal RGB image I/O + drawing for the predict path, backed by the vendored
// stb headers. RgbImage holds packed 8-bit HWC RGB pixels.

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "detr/core/result.hpp"

namespace detr::io {

struct RgbImage {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> data;  // width*height*3, row-major RGB
};

core::Result<RgbImage> LoadRgb(const std::filesystem::path& path);
core::Result<void> SavePng(const std::filesystem::path& path, const RgbImage& img);

// Draws a rectangle outline (corners clamped to the image) of the given color
// and thickness in pixels.
void DrawRect(RgbImage& img, int x0, int y0, int x1, int y1, std::uint8_t r, std::uint8_t g,
              std::uint8_t b, int thickness = 2);

}  // namespace detr::io
