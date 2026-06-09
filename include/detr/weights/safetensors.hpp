// Copyright 2026 detrcpp authors. Apache-2.0.
//
// safetensors reader/writer — the canonical, language-neutral weight format for
// detrcpp. We use it because it is bidirectionally compatible with the original
// PyTorch repos (which can `safetensors.torch.save_file/load_file`) yet trivial
// and safe to parse in C++ with NO Python and NO pickle: the file is
//
//   [ 8 bytes little-endian u64 = header_len ]
//   [ header_len bytes of UTF-8 JSON ]
//   [ tensor data buffer ]
//
// where the JSON maps each tensor name to {dtype, shape, data_offsets:[b,e]}
// (byte range within the data buffer), plus an optional "__metadata__" object
// of string->string. Layout is row-major, little-endian. See
// https://github.com/huggingface/safetensors for the spec.

#pragma once

#include <filesystem>

#include "detr/core/result.hpp"
#include "detr/weights/state_dict.hpp"

namespace detr::weights {

// Loads a .safetensors file into a StateDict. Validates the header length,
// every dtype, and that each tensor's declared byte range matches its
// shape*dtype size and lies within the data buffer.
core::Result<StateDict> LoadSafetensors(const std::filesystem::path& path);

// Writes a StateDict to a .safetensors file. Tensors are laid out in the
// StateDict's insertion order; the "__metadata__" block is written when the
// StateDict carries metadata.
core::Result<void> SaveSafetensors(const std::filesystem::path& path, const StateDict& state);

}  // namespace detr::weights
