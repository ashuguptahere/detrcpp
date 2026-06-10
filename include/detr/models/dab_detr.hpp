// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DAB-DETR (Dynamic Anchor Boxes DETR): queries are 4D anchor boxes (x,y,w,h);
// the decoder uses the shared decoupled cross-attention (cond_decoder.hpp) with
// width/height-modulated positional attention and per-layer iterative anchor
// refinement. Sigmoid/focal head. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>

#include "detr/models/model.hpp"

namespace detr::models {

std::shared_ptr<IModel> MakeDabDetr(const YAML::Node& cfg);
ModelMeta DabDetrMeta(const YAML::Node& cfg);

// DN-DETR: the DAB-DETR network plus denoising training (an extra label_enc and a
// train-only ForwardDenoise); eval/inference is identical to DAB-DETR.
std::shared_ptr<IModel> MakeDnDetr(const YAML::Node& cfg);
ModelMeta DnDetrMeta(const YAML::Node& cfg);

}  // namespace detr::models
