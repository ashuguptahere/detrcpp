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

// Registers the rt-detr / rt-detrv2 / rt-detrv3 family in n/s/m/l/x sizes (plus
// the plain version name as an alias for -l) into the global Registry.
void RegisterRtDetr();

}  // namespace detr::models
