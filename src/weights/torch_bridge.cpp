// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/torch_bridge.hpp"

#include <fmt/format.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace detr::weights {

namespace {

using core::Err;
using core::ErrorCode;
using core::Result;

torch::ScalarType TorchDtype(DType d) {
  switch (d) {
    case DType::F64:
      return torch::kFloat64;
    case DType::F32:
      return torch::kFloat32;
    case DType::F16:
      return torch::kFloat16;
    case DType::BF16:
      return torch::kBFloat16;
    case DType::I64:
      return torch::kInt64;
    case DType::I32:
      return torch::kInt32;
    case DType::I16:
      return torch::kInt16;
    case DType::I8:
      return torch::kInt8;
    case DType::U8:
      return torch::kUInt8;
    case DType::Bool:
      return torch::kBool;
  }
  return torch::kFloat32;
}

Result<DType> FromTorchDtype(torch::ScalarType s) {
  switch (s) {
    case torch::kFloat64:
      return DType::F64;
    case torch::kFloat32:
      return DType::F32;
    case torch::kFloat16:
      return DType::F16;
    case torch::kBFloat16:
      return DType::BF16;
    case torch::kInt64:
      return DType::I64;
    case torch::kInt32:
      return DType::I32;
    case torch::kInt16:
      return DType::I16;
    case torch::kInt8:
      return DType::I8;
    case torch::kUInt8:
      return DType::U8;
    case torch::kBool:
      return DType::Bool;
    default:
      return Err(ErrorCode::Unsupported,
                 fmt::format("unsupported torch dtype: {}", torch::toString(s)));
  }
}

}  // namespace

torch::Tensor ToTensor(const RawTensor& t) {
  std::vector<std::int64_t> shape(t.shape.begin(), t.shape.end());
  torch::Tensor out = torch::empty(shape, torch::TensorOptions().dtype(TorchDtype(t.dtype)));
  if (!t.data.empty()) {
    std::memcpy(out.data_ptr(), t.data.data(), t.data.size());
  }
  return out;
}

Result<RawTensor> FromTensor(const torch::Tensor& t) {
  const torch::Tensor c = t.detach().to(torch::kCPU).contiguous();
  auto dt = FromTorchDtype(c.scalar_type());
  if (!dt) {
    return tl::make_unexpected(dt.error());
  }
  RawTensor r;
  r.dtype = *dt;
  r.shape.assign(c.sizes().begin(), c.sizes().end());
  const auto nbytes =
      static_cast<std::size_t>(c.numel()) * static_cast<std::size_t>(c.element_size());
  r.data.resize(nbytes);
  if (nbytes != 0) {
    std::memcpy(r.data.data(), c.data_ptr(), nbytes);
  }
  return r;
}

StateDict StateDictFromModule(const torch::nn::Module& module) {
  StateDict sd;
  for (const auto& p : module.named_parameters(/*recurse=*/true)) {
    auto r = FromTensor(p.value());
    if (r) {
      sd.Set(p.key(), std::move(*r));
    }
  }
  for (const auto& b : module.named_buffers(/*recurse=*/true)) {
    auto r = FromTensor(b.value());
    if (r) {
      sd.Set(b.key(), std::move(*r));
    }
  }
  return sd;
}

Result<LoadReport> LoadStateDictInto(torch::nn::Module& module, const StateDict& source,
                                     const WeightRemapper& remap, bool strict) {
  torch::NoGradGuard no_grad;
  const StateDict src = remap.Apply(source);

  std::unordered_map<std::string, torch::Tensor> dst;
  for (const auto& p : module.named_parameters(/*recurse=*/true)) {
    dst.emplace(p.key(), p.value());
  }
  for (const auto& b : module.named_buffers(/*recurse=*/true)) {
    dst.emplace(b.key(), b.value());
  }

  LoadReport rep;
  std::unordered_set<std::string> used;
  for (const auto& key : src.Keys()) {
    auto it = dst.find(key);
    if (it == dst.end()) {
      rep.unexpected.push_back(key);
      if (strict) {
        return Err(ErrorCode::InvalidArgument,
                   fmt::format("unexpected key '{}' (no destination in module)", key));
      }
      continue;
    }
    const torch::Tensor want = ToTensor(*src.Find(key));
    if (it->second.sizes() != want.sizes()) {
      rep.mismatched.push_back(key);
      if (strict) {
        return Err(ErrorCode::InvalidArgument, fmt::format("shape mismatch for '{}'", key));
      }
      continue;
    }
    it->second.copy_(want.to(it->second.scalar_type()));
    used.insert(key);
    ++rep.loaded;
  }
  for (const auto& [name, tensor] : dst) {
    if (used.find(name) == used.end()) {
      rep.missing.push_back(name);
    }
  }
  if (strict && !rep.missing.empty()) {
    return Err(ErrorCode::InvalidArgument,
               fmt::format("{} module key(s) not provided by source", rep.missing.size()));
  }
  return rep;
}


}  // namespace detr::weights
