// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Bridge between the framework-independent weight layer (RawTensor / StateDict)
// and live LibTorch modules. Compiled only when DETR_ENABLE_TORCH is on. This is
// where weight interoperability becomes concrete: a StateDict loaded from a
// .pth produced by the original repo is copied, by parameter name, into
// our module — and a trained module is serialized back out to the same
// name space so the original repo can load it.

#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <string>
#include <vector>

#include "detr/core/result.hpp"
#include "detr/weights/remapper.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/tensor.hpp"

namespace detr::weights {

// RawTensor -> torch::Tensor (CPU, owns a copy of the bytes).
[[nodiscard]] torch::Tensor ToTensor(const RawTensor& t);

// torch::Tensor -> RawTensor (forces CPU + contiguous; copies bytes).
[[nodiscard]] core::Result<RawTensor> FromTensor(const torch::Tensor& t);

// A module's parameters and buffers as a StateDict, keyed exactly as
// named_parameters()/named_buffers() (which is what upstream state_dicts use).
[[nodiscard]] StateDict StateDictFromModule(const torch::nn::Module& module);

struct LoadReport {
  std::size_t loaded{0};
  std::vector<std::string> missing;     // module keys not provided by the source
  std::vector<std::string> unexpected;  // source keys with no module destination
  std::vector<std::string> mismatched;  // name matched but shape differed
};

// Copies tensors from |source| (after applying |remap|) into |module| by name.
// With strict=true, a shape mismatch or any missing/unexpected key is an error;
// otherwise they are recorded in the report and loading continues.
[[nodiscard]] core::Result<LoadReport> LoadStateDictInto(torch::nn::Module& module,
                                                         const StateDict& source,
                                                         const WeightRemapper& remap = {},
                                                         bool strict = false);

}  // namespace detr::weights
