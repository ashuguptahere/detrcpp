// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DN-DETR denoising training. From the ground truth, build groups of "denoising
// queries" whose anchor is a noised GT box and whose content is a noised GT
// label; the decoder reconstructs the clean GT (known assignment, no Hungarian).
// This lives in the train layer (it owns the GT) and hands the model tensor-only
// DenoisingInput. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/types.h>

#include <utility>
#include <vector>

#include "detr/models/model.hpp"
#include "detr/train/matcher.hpp"
#include "detr/train/target.hpp"

namespace detr::train {

struct DnConfig {
  int dn_number{5};            // denoising groups per image
  double box_noise_scale{0.4};
  double label_noise_ratio{0.2};
  double weight{1.0};  // multiplier on the whole DN loss contribution
};

// Per-group layout used to build the known reconstruction assignment.
struct DnLayout {
  torch::Tensor tgt_index;  // [B, num_dn] int64, GT index per slot, -1 for padding
  torch::Tensor pad_mask;   // [B, num_dn] bool, true = a valid (non-padding) slot
  int num_dn{0};
};

// Builds the model's denoising input (noised anchors + labels + the group-isolation
// attention mask) and the layout, from the GT batch. num_dn = dn_number * max-T,
// group-major with stride max-T so one [L,L] mask is valid for the whole batch.
// Returns active=false when there are no GT objects. Runs under NoGrad on CPU.
std::pair<models::DenoisingInput, DnLayout> MakeDenoising(const TargetBatch& targets,
                                                          const DnConfig& cfg, int num_classes,
                                                          int num_queries);

// The known reconstruction assignment (one MatchIndices per image): each valid
// denoising query maps to the original (clean) GT it must reconstruct.
std::vector<MatchIndices> BuildDnMatches(const DnLayout& layout);

}  // namespace detr::train
