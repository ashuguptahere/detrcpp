// Copyright 2026 detrcpp authors. Apache-2.0.
//
// A faithful, dependency-free reimplementation of COCO's detection metric
// (pycocotools COCOeval): AP averaged over IoU thresholds [0.5:0.05:0.95] with
// 101-point recall interpolation, plus AP50/AP75, the small/medium/large area
// breakdown (the small-object metric DETR papers report), and average recall at
// {1,10,100} detections. Pure C++ on boxes + scores, so it is unit-testable
// without LibTorch.

#pragma once

#include <vector>

namespace detr::eval {

// All boxes are absolute pixel xywh (COCO convention). |area| is the COCO
// annotation area (segmentation area) used for the small/medium/large bucketing,
// matching pycocotools; 0 means "fall back to the bbox area w*h".
struct GtBox {
  int category_id{0};
  float x{0};
  float y{0};
  float w{0};
  float h{0};
  float area{0};
  bool iscrowd{false};
};

struct DtBox {
  int category_id{0};
  float x{0};
  float y{0};
  float w{0};
  float h{0};
  float score{0};
};

// One image's ground truth + detections.
struct EvalImage {
  std::vector<GtBox> gts;
  std::vector<DtBox> dts;
};

// The 12 standard COCO summary numbers (each in [0,1], or -1 if undefined, e.g.
// an area range with no ground-truth boxes).
struct CocoMetrics {
  double ap{-1};         // AP @ [.5:.95]   (the primary metric)
  double ap50{-1};       // AP @ .50
  double ap75{-1};       // AP @ .75
  double ap_small{-1};   // AP, area < 32^2
  double ap_medium{-1};  // AP, 32^2 <= area < 96^2
  double ap_large{-1};   // AP, area >= 96^2
  double ar1{-1};        // AR given 1 detection / image
  double ar10{-1};       // AR given 10
  double ar100{-1};      // AR given 100
  double ar_small{-1};
  double ar_medium{-1};
  double ar_large{-1};
};

// Evaluates detections against ground truth over the given category ids.
CocoMetrics CocoEvaluate(const std::vector<EvalImage>& images,
                         const std::vector<int>& category_ids);

}  // namespace detr::eval
