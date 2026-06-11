// Copyright 2026 detrcpp authors. Apache-2.0.
//
// SAHI (Slicing Aided Hyper Inference): slice a large image into overlapping
// tiles, detect per tile, shift each tile's boxes back to full-image coordinates,
// and merge across tiles with class-aware NMS (or NMM). Improves small-object
// recall on high-resolution inputs. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <functional>
#include <vector>

#include "detr/eval/coco_eval.hpp"
#include "detr/io/image.hpp"
#include "detr/models/model.hpp"

namespace detr::infer {

struct SahiConfig {
  int slice_h{512};
  int slice_w{512};
  double overlap_h{0.2};        // fraction of slice_h overlapped between rows
  double overlap_w{0.2};        // fraction of slice_w overlapped between cols
  bool merge_full_image{false};  // also run one full-image pass (standard SAHI)
  float postprocess_iou{0.5F};   // cross-tile merge IoU threshold
  float conf{0.0F};              // per-tile score floor before merging
  enum class Merge { kNms, kNmm };
  Merge merge{Merge::kNms};
};

// Pixel box of a tile within the full image (corner form, half-open [x0,x1)).
struct Slice {
  int x0{0};
  int y0{0};
  int x1{0};
  int y1{0};
};

// Per-tile detector: given an RGB crop, returns DtBox in crop-local absolute xywh.
using TileDetector = std::function<std::vector<eval::DtBox>(const io::RgbImage&)>;

// The overlapping tile grid (exposed for testing the tiling math in isolation).
std::vector<Slice> ComputeSlices(int img_w, int img_h, const SahiConfig& cfg);

// Class-aware greedy NMS (the repo's first), built on train::BoxIou. With
// merge=true it instead folds suppressed boxes into the kept one (NMM).
std::vector<eval::DtBox> NmsPerClass(std::vector<eval::DtBox> dets, float iou_thr,
                                     bool merge = false);

// Run |detect| over the tiles, shift boxes back, and merge. Returns full-image
// absolute-pixel xywh DtBox (the same convention as PostprocessImage).
std::vector<eval::DtBox> SahiDetect(const io::RgbImage& image, const TileDetector& detect,
                                    const SahiConfig& cfg);

// Convenience overload that builds the per-tile detector from a model (runs the
// model's preprocess -> Forward -> postprocess on the model's device).
std::vector<eval::DtBox> SahiDetect(models::IModel& model, const io::RgbImage& image,
                                    const SahiConfig& cfg);

}  // namespace detr::infer
