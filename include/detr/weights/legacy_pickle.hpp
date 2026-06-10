// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Loader for legacy (pre-PyTorch-1.6) torch.save checkpoints: a raw protocol-2
// pickle of an OrderedDict[str -> Tensor] followed by length-prefixed
// little-endian storage payloads. No Python, no LibTorch — a minimal pickle VM
// that recognizes only the globals a torch/torchvision state_dict emits
// (_rebuild_tensor_v2, _rebuild_parameter, OrderedDict, the storage classes) and
// returns Unsupported on anything richer. Lives in the torch-free weights lib.

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

#include "detr/core/result.hpp"
#include "detr/weights/state_dict.hpp"

namespace detr::weights {

// Parse a legacy .pth file into a StateDict. Returns Unsupported (never a wrong
// tensor) on any construct outside the recognized state_dict subset.
[[nodiscard]] core::Result<StateDict> LoadLegacyPth(const std::filesystem::path& path);

// Same, from an in-memory buffer (used by tests with crafted/round-tripped bytes).
[[nodiscard]] core::Result<StateDict> LoadLegacyPthBytes(std::span<const std::byte> bytes);

}  // namespace detr::weights
