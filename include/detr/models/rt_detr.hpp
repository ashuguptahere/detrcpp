// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RT-DETR (Real-Time DETR): ResNet backbone -> hybrid encoder (AIFI attention on
// the top level + CNN cross-scale fusion / CCFM) -> IoU-aware query selection ->
// deformable decoder with iterative box refinement. Sigmoid/focal head. Reuses
// the shared ResNet (resnet.hpp) + MSDeformAttn (deform_attn.hpp). Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>

#include "detr/models/model.hpp"

namespace detr::models {

std::shared_ptr<IModel> MakeRtDetr(const YAML::Node& cfg);
ModelMeta RtDetrMeta(const YAML::Node& cfg);

}  // namespace detr::models
