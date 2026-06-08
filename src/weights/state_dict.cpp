// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/state_dict.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace detr::weights {

void StateDict::Set(std::string name, RawTensor tensor) {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    order_.push_back(name);
  }
  tensors_[std::move(name)] = std::move(tensor);
}

const RawTensor* StateDict::Find(std::string_view name) const {
  auto it = tensors_.find(std::string(name));
  return it == tensors_.end() ? nullptr : &it->second;
}

bool StateDict::Contains(std::string_view name) const {
  return tensors_.find(std::string(name)) != tensors_.end();
}

void StateDict::SetMeta(std::string key, std::string value) {
  meta_[std::move(key)] = std::move(value);
}

std::optional<std::string> StateDict::GetMeta(std::string_view key) const {
  auto it = meta_.find(std::string(key));
  if (it == meta_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::size_t StateDict::TotalBytes() const {
  std::size_t total = 0;
  for (const auto& name : order_) {
    total += tensors_.at(name).data.size();
  }
  return total;
}

}  // namespace detr::weights
