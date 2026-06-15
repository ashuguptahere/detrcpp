// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DETR (DEtection TRansformer) — a conv backbone feeding a transformer
// encoder/decoder with learned object queries and class/box heads. This is the
// canonical end-to-end set-prediction detector; it trains from scratch and its
// weights serialize to .pth for interchange. Compiled with
// DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>

#include "detr/models/model.hpp"

namespace detr::models {

// Builds a DETR model. Recognized config keys (all optional, with DETR defaults):
//   hidden_dim(256) nheads(8) enc_layers(6) dec_layers(6) dim_feedforward(2048)
//   num_queries(100) num_classes(91) imgsz(640) backbone_width(64).
std::shared_ptr<IModel> MakeDetr(const YAML::Node& cfg);

ModelMeta DetrMeta(const YAML::Node& cfg);

}  // namespace detr::models
