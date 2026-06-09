// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DETR with a torchvision-style ResNet-50 backbone (the canonical DETR). Reuses
// the shared transformer head (detr_head.hpp); only the backbone differs from
// the compact `detr`. Demonstrates backbone modularity end to end: train, predict,
// eval, and ONNX export. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <memory>

#include <yaml-cpp/yaml.h>

#include "detr/models/model.hpp"

namespace detr::models {

std::shared_ptr<IModel> MakeDetrR50(const YAML::Node& cfg);
ModelMeta DetrR50Meta(const YAML::Node& cfg);

// detr-r101: same as detr-r50 with a ResNet-101 backbone ({3,4,23,3} blocks).
std::shared_ptr<IModel> MakeDetrR101(const YAML::Node& cfg);
ModelMeta DetrR101Meta(const YAML::Node& cfg);

}  // namespace detr::models
