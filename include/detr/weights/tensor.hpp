// Copyright 2026 detrcpp authors. Apache-2.0.
//
// RawTensor: a LibTorch-independent owner of a tensor's dtype, shape, and raw
// little-endian bytes. The weight-interchange layer (safetensors, .pth) speaks
// in RawTensor so it can be built and tested with no LibTorch dependency; a thin
// optional bridge (detr/weights/torch_bridge.hpp, compiled only with
// DETR_ENABLE_TORCH) converts RawTensor <-> torch::Tensor.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "detr/core/result.hpp"

namespace detr::weights {

// dtypes named exactly as the safetensors spec encodes them.
enum class DType {
  F64,
  F32,
  F16,
  BF16,
  I64,
  I32,
  I16,
  I8,
  U8,
  Bool,
};

// Size of one element in bytes.
constexpr std::size_t DTypeSize(DType d) {
  switch (d) {
    case DType::F64:
    case DType::I64:
      return 8;
    case DType::F32:
    case DType::I32:
      return 4;
    case DType::F16:
    case DType::BF16:
    case DType::I16:
      return 2;
    case DType::I8:
    case DType::U8:
    case DType::Bool:
      return 1;
  }
  return 0;
}

// safetensors dtype string, e.g. DType::F32 -> "F32".
std::string_view DTypeName(DType d);

// Parse a safetensors dtype string. Returns InvalidArgument on unknown names.
core::Result<DType> DTypeFromName(std::string_view name);

struct RawTensor {
  DType dtype{DType::F32};
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;  // raw little-endian element bytes, row-major.

  std::int64_t Numel() const {
    std::int64_t n = 1;
    for (const std::int64_t d : shape) {
      n *= d;
    }
    return shape.empty() ? 0 : n;  // scalar (rank-0) has numel 1; empty shape -> 0.
  }

  std::size_t Nbytes() const {
    return static_cast<std::size_t>(Numel() < 0 ? 0 : Numel()) * DTypeSize(dtype);
  }
};

}  // namespace detr::weights
