// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/data/loader.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "stb_image.h"
#include "stb_image_resize2.h"

#include "detr/log/log.hpp"

namespace detr::data {

namespace {

// Decodes |path|, resizes to imgsz x imgsz, returns a normalized [3,H,W] float
// tensor, or an undefined tensor on failure.
torch::Tensor DecodeImage(const std::string& path, int imgsz) {
  int w = 0;
  int h = 0;
  int comp = 0;
  unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &comp, 3);
  if (pixels == nullptr || w <= 0 || h <= 0) {
    if (pixels != nullptr) {
      stbi_image_free(pixels);
    }
    return {};
  }
  std::vector<unsigned char> resized(static_cast<std::size_t>(imgsz) *
                                     static_cast<std::size_t>(imgsz) * 3);
  stbir_resize_uint8_linear(pixels, w, h, 0, resized.data(), imgsz, imgsz, 0, STBIR_RGB);
  stbi_image_free(pixels);

  auto hwc = torch::from_blob(resized.data(), {imgsz, imgsz, 3}, torch::kUInt8).clone();
  auto chw = hwc.to(torch::kFloat32).div_(255.0).permute({2, 0, 1}).contiguous();
  static const auto mean = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
  static const auto stddev = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
  return (chw - mean) / stddev;
}

train::Target TargetFromSample(const Sample& s) {
  train::Target t;
  const auto n = static_cast<std::int64_t>(s.boxes.size());
  t.labels = torch::empty({n}, torch::kInt64);
  t.boxes = torch::empty({n, 4}, torch::kFloat32);
  auto la = t.labels.accessor<std::int64_t, 1>();
  auto ba = t.boxes.accessor<float, 2>();
  for (std::int64_t i = 0; i < n; ++i) {
    const auto& b = s.boxes[static_cast<std::size_t>(i)];
    la[i] = b.class_id;
    ba[i][0] = b.cx;
    ba[i][1] = b.cy;
    ba[i][2] = b.w;
    ba[i][3] = b.h;
  }
  return t;
}

}  // namespace

DataLoader::DataLoader(Dataset dataset, Split split, int imgsz, int batch_size, std::uint64_t seed)
    : dataset_(std::move(dataset)), imgsz_(imgsz), batch_(batch_size) {
  indices_ = dataset_.IndicesOf(split);
  Reshuffle(seed);
}

std::size_t DataLoader::NumBatches() const {
  if (batch_ <= 0 || indices_.empty()) {
    return 0;
  }
  const auto b = static_cast<std::size_t>(batch_);
  return (indices_.size() + b - 1) / b;  // ceil; last batch may be partial.
}

void DataLoader::Reshuffle(std::uint64_t epoch_seed) {
  // Canonicalize by sample index, then Fisher–Yates with mt19937_64 — the same
  // reproducible scheme Dataset::Shuffle uses.
  std::sort(indices_.begin(), indices_.end());
  std::mt19937_64 rng(epoch_seed);
  for (std::size_t i = indices_.size(); i > 1; --i) {
    const std::uint64_t j = rng() % i;
    std::swap(indices_[i - 1], indices_[static_cast<std::size_t>(j)]);
  }
}

core::Result<Batch> DataLoader::At(std::size_t i) const {
  if (i >= NumBatches()) {
    return core::Err(core::ErrorCode::InvalidArgument,
                     fmt::format("batch index {} out of range ({})", i, NumBatches()));
  }
  const auto b = static_cast<std::size_t>(batch_);
  const std::size_t begin = i * b;
  const std::size_t end = std::min(begin + b, indices_.size());

  std::vector<torch::Tensor> imgs;
  Batch out;
  for (std::size_t k = begin; k < end; ++k) {
    const Sample& s = dataset_.samples[indices_[k]];
    auto img = DecodeImage(s.image_path, imgsz_);
    if (!img.defined()) {
      detr::log::Get("data.loader").warn("could not decode '{}'; using blank", s.image_path);
      img = torch::zeros({3, imgsz_, imgsz_});
      train::Target empty;
      empty.labels = torch::empty({0}, torch::kInt64);
      empty.boxes = torch::empty({0, 4}, torch::kFloat32);
      out.targets.push_back(std::move(empty));
    } else {
      out.targets.push_back(TargetFromSample(s));
    }
    imgs.push_back(std::move(img));
    out.sizes.emplace_back(s.width, s.height);
    out.sample_indices.push_back(indices_[k]);
  }
  out.images = torch::stack(imgs, 0);
  return out;
}

}  // namespace detr::data
