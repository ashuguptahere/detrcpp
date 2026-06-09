// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DINO: a multi-scale deformable encoder (Deformable-DETR) + IoU-aware query
// selection + a deformable decoder with iterative box refinement. Reuses the
// shared ResNet + MSDeformAttn + focal path. (Contrastive denoising training is a
// tracked follow-up; this is the inference architecture.) Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>

#include "detr/models/model.hpp"

namespace detr::models {

std::shared_ptr<IModel> MakeDino(const YAML::Node& cfg);
ModelMeta DinoMeta(const YAML::Node& cfg);

}  // namespace detr::models
