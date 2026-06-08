// Copyright 2026 detrcpp authors. Apache-2.0.
//
// StateDict: an insertion-ordered map of parameter-name -> RawTensor, plus
// optional string metadata. This is the in-memory form of a model's weights,
// independent of any framework. Loaders (safetensors, .pth) produce one;
// savers consume one; the torch bridge converts it to/from a live module.
//
// Insertion order is preserved so that round-tripping weights is deterministic
// and diffable, and so a saved file lists tensors in a stable order.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "detr/weights/tensor.hpp"

namespace detr::weights {

class StateDict {
 public:
  StateDict() = default;

  // Inserts or replaces. New keys are appended to the order; replacing an
  // existing key keeps its position.
  void Set(std::string name, RawTensor tensor);

  const RawTensor* Find(std::string_view name) const;
  bool Contains(std::string_view name) const;

  // Ordered list of tensor names (insertion order).
  const std::vector<std::string>& Keys() const { return order_; }
  std::size_t Size() const { return order_.size(); }
  bool Empty() const { return order_.empty(); }

  // String metadata (round-trips through safetensors' "__metadata__").
  void SetMeta(std::string key, std::string value);
  std::optional<std::string> GetMeta(std::string_view key) const;
  const std::map<std::string, std::string>& Meta() const { return meta_; }

  // Total bytes of tensor payloads (excludes header/metadata).
  std::size_t TotalBytes() const;

 private:
  std::vector<std::string> order_;
  std::unordered_map<std::string, RawTensor> tensors_;
  std::map<std::string, std::string> meta_;
};

}  // namespace detr::weights
