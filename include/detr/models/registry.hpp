// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Model registry: the single extension point for adding DETR variants. A model
// is added by registering a name + ModelMeta + factory; nothing else in the
// codebase needs to change (SOLID open/closed). Build() constructs a model by
// name from an optional YAML config. Compiled only with DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "detr/core/result.hpp"
#include "detr/models/model.hpp"

namespace detr::models {

class Registry {
 public:
  using Factory = std::function<std::shared_ptr<IModel>(const YAML::Node&)>;

  static Registry& Instance();

  void Register(std::string name, ModelMeta meta, Factory factory);

  core::Result<std::shared_ptr<IModel>> Build(std::string_view name,
                                              const YAML::Node& cfg = {}) const;

  // Metadata for every registered model, sorted by name.
  std::vector<ModelMeta> List() const;
  bool Contains(std::string_view name) const;

 private:
  struct Entry {
    ModelMeta meta;
    Factory factory;
  };
  mutable std::mutex mu_;
  std::map<std::string, Entry, std::less<>> entries_;
};

// Registers all built-in model variants. Called once at startup (explicit, so
// there is no reliance on static-init ordering or whole-archive linking).
void RegisterBuiltins();

}  // namespace detr::models
