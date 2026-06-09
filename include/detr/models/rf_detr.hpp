// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RF-DETR (Roboflow DETR): a ViT (DINOv2-style) backbone projected to multiple
// scales, IoU-aware query selection, and a deformable decoder with iterative box
// refinement. Reuses the ViT backbone (vit.hpp) + MSDeformAttn (deform_attn.hpp).
// Sigmoid/focal head. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>

#include "detr/models/model.hpp"

namespace detr::models {

std::shared_ptr<IModel> MakeRfDetr(const YAML::Node& cfg);
ModelMeta RfDetrMeta(const YAML::Node& cfg);

}  // namespace detr::models
