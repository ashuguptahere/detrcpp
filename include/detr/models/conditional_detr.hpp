// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Conditional-DETR: DETR with a conditional decoder cross-attention that
// decouples content and spatial queries (the spatial query is a learned
// transformation of the reference-point sine embedding). Reuses the shared
// ResNet; a standard encoder; a custom decoder. Sigmoid/focal head. Compiled
// with DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>

#include "detr/models/model.hpp"

namespace detr::models {

std::shared_ptr<IModel> MakeConditionalDetr(const YAML::Node& cfg);
ModelMeta ConditionalDetrMeta(const YAML::Node& cfg);

}  // namespace detr::models
