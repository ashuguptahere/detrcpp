// Copyright 2026 detrcpp authors. Apache-2.0.
//
// The DETR transformer head, shared by every DETR-family backbone variant: sine
// positional encoding, the multi-head-attention encoder/decoder, learned object
// queries, and the class/box prediction heads. BuildDetrHead registers the
// submodules directly into the owning model (flat names: encoder.*, decoder.*,
// query_embed, class_embed, bbox_embed) so a given variant's parameter names do
// not depend on which backbone it uses. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include "detr/models/model.hpp"

namespace detr::models {

struct DetrConfig {
  int hidden_dim{256};
  int nheads{8};
  int enc_layers{6};
  int dec_layers{6};
  int dim_feedforward{2048};
  int num_queries{100};
  int num_classes{91};
};

// Handles to the head submodules (owned by the model that registered them).
struct DetrHead {
  torch::nn::Embedding query_embed{nullptr};
  torch::nn::ModuleList encoder{nullptr};
  torch::nn::ModuleList decoder{nullptr};
  torch::nn::Linear class_embed{nullptr};
  torch::nn::Sequential bbox_embed{nullptr};
};

// Registers the head into |model| and returns the handles.
DetrHead BuildDetrHead(torch::nn::Module& model, const DetrConfig& cfg);

// Runs the transformer + heads on a projected backbone feature map
// |src| of shape [B, hidden_dim, h, w]. Returns DETR's logits + boxes.
Detections RunDetrHead(const DetrHead& head, torch::Tensor src, const DetrConfig& cfg);

}  // namespace detr::models
