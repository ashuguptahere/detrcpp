// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Deformable-DETR (single-stage): multi-scale ResNet features + multi-scale
// deformable encoder/decoder (MSDeformAttn) + learned queries with reference
// points. Reuses the shared ResNet (resnet.hpp) and deformable attention
// (deform_attn.hpp). Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <memory>

#include <yaml-cpp/yaml.h>

#include "detr/models/model.hpp"

namespace detr::models {

std::shared_ptr<IModel> MakeDeformableDetr(const YAML::Node& cfg);
ModelMeta DeformableDetrMeta(const YAML::Node& cfg);

}  // namespace detr::models
