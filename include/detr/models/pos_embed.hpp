// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Shared 2D sine positional embedding (DETR's PositionEmbeddingSine, no padding
// mask), hoisted here so the DETR-family heads share one definition instead of
// duplicating it per variant. It is parameter-free, so this consolidation does
// not touch any weight layout. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/types.h>

#include <cstdint>

namespace detr::models {

// Returns a [B, d, h, w] sine/cosine positional encoding on |opts|'s device.
torch::Tensor SinePos(std::int64_t b, std::int64_t d, std::int64_t h, std::int64_t w,
                      const torch::TensorOptions& opts);

}  // namespace detr::models
