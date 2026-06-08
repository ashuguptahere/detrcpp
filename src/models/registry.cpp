// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/registry.hpp"

#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace detr::models {

Registry& Registry::Instance() {
  static Registry instance;
  return instance;
}

void Registry::Register(std::string name, ModelMeta meta, Factory factory) {
  std::lock_guard<std::mutex> lock(mu_);
  entries_[std::move(name)] = Entry{std::move(meta), std::move(factory)};
}

core::Result<std::shared_ptr<IModel>> Registry::Build(std::string_view name,
                                                      const YAML::Node& cfg) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(name);
  if (it == entries_.end()) {
    return core::Err(core::ErrorCode::NotFound,
                     fmt::format("unknown model '{}' (see --list-models)", name));
  }
  return it->second.factory(cfg);
}

std::vector<ModelMeta> Registry::List() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<ModelMeta> out;
  out.reserve(entries_.size());
  for (const auto& [name, entry] : entries_) {
    out.push_back(entry.meta);
  }
  return out;
}

bool Registry::Contains(std::string_view name) const {
  std::lock_guard<std::mutex> lock(mu_);
  return entries_.find(name) != entries_.end();
}

}  // namespace detr::models
