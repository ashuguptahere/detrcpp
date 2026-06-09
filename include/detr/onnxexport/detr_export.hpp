// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Emits a full DETR forward pass as a fixed-shape ONNX graph (batch 1), mirroring
// models::DetrImpl::Forward exactly: conv backbone, input projection, sine
// positional encoding (baked as a constant), transformer encoder/decoder with
// multi-head attention decomposed into MatMul/Softmax/Transpose/Reshape, learned
// object queries, and the class/box heads. Weights come from a StateDict (loaded
// from .safetensors) — no LibTorch, no Python. Compiled with DETR_ENABLE_ONNX.

#pragma once

#include <string>

#include "detr/core/result.hpp"
#include "detr/weights/state_dict.hpp"

namespace detr::onnxexport {

enum class Backbone { Compact, ResNet50 };

// Mirrors the model's Config (the architecture the weights belong to).
struct DetrArch {
  Backbone backbone{Backbone::Compact};
  int hidden_dim{256};
  int nheads{8};
  int enc_layers{6};
  int dec_layers{6};
  int dim_feedforward{2048};
  int num_queries{100};
  int num_classes{91};
  int imgsz{640};
  int backbone_width{64};  // Compact backbone only
};

// Builds the ONNX model from |arch| + |weights| and writes it to |path| (after
// onnx::checker validation). Inputs:  "images" [1,3,imgsz,imgsz].
// Outputs: "logits" [1,num_queries,num_classes+1], "boxes" [1,num_queries,4].
core::Result<void> ExportDetr(const DetrArch& arch, const weights::StateDict& weights,
                              const std::string& path);

}  // namespace detr::onnxexport
