// Copyright 2026 detrcpp authors. Apache-2.0.
//
// IModel: the interface every DETR variant implements. It is a torch::nn::Module
// so the optimizer, EMA, checkpointing, and the weight bridge all operate on it
// uniformly. Forward returns DETR's two prediction heads (class logits + boxes).
// Compiled only with DETR_ENABLE_TORCH.

#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>

#include <string>
#include <vector>

#include "detr/weights/remapper.hpp"

namespace detr::models {

struct ModelMeta {
  std::string name;
  int imgsz{640};
  int num_classes{91};  // COCO categories (the +1 no-object slot is internal)
  int num_queries{100};
  // Classification mode. false: softmax over num_classes+1 (DETR, with a no-object
  // slot). true: sigmoid/focal over num_classes (Deformable-DETR and the modern
  // variants — no no-object slot). Drives the criterion, matcher, and postprocess.
  bool focal{false};
  std::string license{"Apache-2.0"};
  std::string upstream;  // provenance URL for the architecture / weights
};

// DETR's set-prediction output. logits: [B, Q, num_classes (+1 for softmax
// models, where the last is no-object)]; boxes: [B, Q, 4] normalized cxcywh.
struct Detections {
  torch::Tensor logits;
  torch::Tensor boxes;

  // Auxiliary per-decoder-layer predictions for DETR's deep supervision. Each
  // aux_logits[i]/aux_boxes[i] mirrors logits/boxes for an intermediate decoder
  // layer. Populated by the heads only in training mode (empty at inference, so
  // postprocess/eval are unaffected); the trainer adds a set loss per entry.
  std::vector<torch::Tensor> aux_logits;
  std::vector<torch::Tensor> aux_boxes;
};

class IModel : public torch::nn::Module {
 public:
  ~IModel() override = default;

  // images: [B, 3, H, W], normalized. Returns per-query class + box predictions.
  virtual Detections Forward(torch::Tensor images) = 0;

  virtual ModelMeta Meta() const = 0;

  // Rules mapping an upstream checkpoint's parameter names onto this module's
  // names, so the original repo's weights load 1:1 without touching the weights
  // or the module tree. Identity by default.
  virtual weights::WeightRemapper UpstreamRemapper() const { return {}; }
};

}  // namespace detr::models
