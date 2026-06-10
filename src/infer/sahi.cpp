// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/infer/sahi.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <map>

#include "detr/infer/postprocess.hpp"
#include "detr/infer/preprocess.hpp"
#include "detr/train/box_ops.hpp"

namespace detr::infer {

namespace {

// Overlapping origins along one axis. step = slice - round(overlap*slice); the
// last tile is back-shifted to sit flush with the image border (no short tiles).
std::vector<int> AxisOrigins(int img, int slice, double overlap) {
  std::vector<int> origins;
  if (img <= slice) {
    origins.push_back(0);
    return origins;
  }
  const int step = std::max(1, slice - static_cast<int>(std::lround(overlap * slice)));
  for (int x = 0;; x += step) {
    int start = x;
    const bool last = (start + slice >= img);
    if (last) {
      start = img - slice;
    }
    if (origins.empty() || origins.back() != start) {
      origins.push_back(start);
    }
    if (last) {
      break;
    }
  }
  return origins;
}

io::RgbImage Crop(const io::RgbImage& img, const Slice& s) {
  io::RgbImage out;
  out.width = s.x1 - s.x0;
  out.height = s.y1 - s.y0;
  out.data.resize(static_cast<std::size_t>(out.width) * static_cast<std::size_t>(out.height) * 3);
  for (int y = 0; y < out.height; ++y) {
    const std::size_t src = (static_cast<std::size_t>(y + s.y0) * static_cast<std::size_t>(img.width) +
                             static_cast<std::size_t>(s.x0)) * 3;
    const std::size_t dst = static_cast<std::size_t>(y) * static_cast<std::size_t>(out.width) * 3;
    std::copy_n(img.data.begin() + static_cast<std::ptrdiff_t>(src),
                static_cast<std::size_t>(out.width) * 3, out.data.begin() + static_cast<std::ptrdiff_t>(dst));
  }
  return out;
}

}  // namespace

std::vector<Slice> ComputeSlices(int img_w, int img_h, const SahiConfig& cfg) {
  const auto xs = AxisOrigins(img_w, cfg.slice_w, cfg.overlap_w);
  const auto ys = AxisOrigins(img_h, cfg.slice_h, cfg.overlap_h);
  std::vector<Slice> slices;
  slices.reserve(xs.size() * ys.size());
  for (const int y0 : ys) {
    for (const int x0 : xs) {
      slices.push_back(Slice{x0, y0, std::min(x0 + cfg.slice_w, img_w),
                             std::min(y0 + cfg.slice_h, img_h)});
    }
  }
  return slices;
}

std::vector<eval::DtBox> NmsPerClass(std::vector<eval::DtBox> dets, float iou_thr, bool merge) {
  torch::NoGradGuard ng;
  std::map<int, std::vector<std::size_t>> by_cls;
  for (std::size_t i = 0; i < dets.size(); ++i) {
    by_cls[dets[i].category_id].push_back(i);
  }
  std::vector<eval::DtBox> out;
  for (auto& [cls, idx] : by_cls) {
    (void)cls;
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return dets[a].score > dets[b].score; });
    const auto n = static_cast<std::int64_t>(idx.size());
    auto xyxy = torch::empty({n, 4});
    auto acc = xyxy.accessor<float, 2>();
    for (std::int64_t k = 0; k < n; ++k) {
      const auto& d = dets[idx[static_cast<std::size_t>(k)]];
      acc[k][0] = d.x;
      acc[k][1] = d.y;
      acc[k][2] = d.x + d.w;
      acc[k][3] = d.y + d.h;
    }
    auto iou = train::BoxIou(xyxy, xyxy).first.contiguous();  // [n, n]
    auto iacc = iou.accessor<float, 2>();
    std::vector<char> removed(idx.size(), 0);
    for (std::int64_t k = 0; k < n; ++k) {
      if (removed[static_cast<std::size_t>(k)]) {
        continue;
      }
      eval::DtBox box = dets[idx[static_cast<std::size_t>(k)]];
      for (std::int64_t m = k + 1; m < n; ++m) {
        if (removed[static_cast<std::size_t>(m)] || iacc[k][m] <= iou_thr) {
          continue;
        }
        removed[static_cast<std::size_t>(m)] = 1;
        if (merge) {  // NMM: grow the kept box to the union of the corners
          const auto& o = dets[idx[static_cast<std::size_t>(m)]];
          const float x2 = std::max(box.x + box.w, o.x + o.w);
          const float y2 = std::max(box.y + box.h, o.y + o.h);
          box.x = std::min(box.x, o.x);
          box.y = std::min(box.y, o.y);
          box.w = x2 - box.x;
          box.h = y2 - box.y;
        }
      }
      out.push_back(box);
    }
  }
  return out;
}

std::vector<eval::DtBox> SahiDetect(const io::RgbImage& image, const TileDetector& detect,
                                    const SahiConfig& cfg) {
  std::vector<eval::DtBox> pool;
  for (const Slice& s : ComputeSlices(image.width, image.height, cfg)) {
    auto local = detect(Crop(image, s));
    for (auto& d : local) {
      if (d.score < cfg.conf) {
        continue;
      }
      d.x += static_cast<float>(s.x0);  // shift back to full-image coords
      d.y += static_cast<float>(s.y0);
      pool.push_back(d);
    }
  }
  if (cfg.merge_full_image) {
    for (auto& d : detect(image)) {
      if (d.score >= cfg.conf) {
        pool.push_back(d);
      }
    }
  }
  return NmsPerClass(std::move(pool), cfg.postprocess_iou, cfg.merge == SahiConfig::Merge::kNmm);
}

std::vector<eval::DtBox> SahiDetect(models::IModel& model, const io::RgbImage& image,
                                    const SahiConfig& cfg) {
  const auto meta = model.Meta();
  torch::Device device(torch::kCPU);
  for (const auto& p : model.parameters()) {
    device = p.device();
    break;
  }
  model.eval();
  TileDetector detect = [&](const io::RgbImage& tile) {
    torch::NoGradGuard ng;
    auto input = PreprocessImage(tile, meta.imgsz, meta.imagenet_norm).to(device);
    auto outputs = model.Forward(input);
    return PostprocessImage(outputs, 0, tile.width, tile.height, meta.num_classes, meta.focal);
  };
  return SahiDetect(image, detect, cfg);
}

}  // namespace detr::infer
