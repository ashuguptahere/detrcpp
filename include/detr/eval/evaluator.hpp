// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Runs a model over a dataset split and returns COCO metrics. Ground-truth boxes
// are taken in absolute pixels from each Sample's original (width, height), so
// the area-based small/medium/large breakdown is meaningful. Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include "detr/data/dataset.hpp"
#include "detr/eval/coco_eval.hpp"
#include "detr/models/model.hpp"

namespace detr::eval {

CocoMetrics EvaluateModel(models::IModel& model, const data::Dataset& dataset,
                          data::Split split, int imgsz, int batch, torch::Device device);

}  // namespace detr::eval
