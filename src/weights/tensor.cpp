// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/tensor.hpp"

#include <array>
#include <string_view>
#include <utility>

namespace detr::weights {

namespace {

constexpr std::array<std::pair<DType, std::string_view>, 10> kNames{{
    {DType::F64, "F64"},
    {DType::F32, "F32"},
    {DType::F16, "F16"},
    {DType::BF16, "BF16"},
    {DType::I64, "I64"},
    {DType::I32, "I32"},
    {DType::I16, "I16"},
    {DType::I8, "I8"},
    {DType::U8, "U8"},
    {DType::Bool, "BOOL"},
}};

}  // namespace

std::string_view DTypeName(DType d) {
  for (const auto& [k, v] : kNames) {
    if (k == d) {
      return v;
    }
  }
  return "F32";
}

core::Result<DType> DTypeFromName(std::string_view name) {
  for (const auto& [k, v] : kNames) {
    if (v == name) {
      return k;
    }
  }
  return core::Err(core::ErrorCode::Unsupported,
                   std::string("unsupported safetensors dtype: ").append(name));
}

}  // namespace detr::weights
