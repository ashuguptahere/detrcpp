// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Torch-free reader/writer for PyTorch `.pth` checkpoints — the detrcpp weight
// interchange format. Reading dispatches on the file's first bytes: a "PK" zip is a
// modern `torch.save` archive (read via miniz + the shared restricted pickle VM); a
// 0x80 pickle is the legacy (pre-1.6) format (delegated to LoadLegacyPth). The writer
// emits the modern `torch.save` layout (a dict of `_rebuild_tensor_v2` tensors + raw
// storages), so the output is loadable by both detrcpp and Python `torch.load`. No
// LibTorch and no Python — usable from the ONNX exporter and the tests too.

#pragma once

#include <filesystem>

#include "detr/core/result.hpp"
#include "detr/weights/state_dict.hpp"

namespace detr::weights {

// Loads a .pth (modern zip or legacy pickle) into a StateDict. Safe on untrusted
// files: the pickle VM only recognizes tensor-rebuild constructs (no code execution).
[[nodiscard]] core::Result<StateDict> LoadPth(const std::filesystem::path& path);

// Writes a StateDict as a modern torch.save .pth (uncompressed zip container).
[[nodiscard]] core::Result<void> SavePth(const std::filesystem::path& path, const StateDict& state);

}  // namespace detr::weights
