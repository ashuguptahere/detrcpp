// Copyright 2026 detrcpp authors. Apache-2.0.
//
// WeightRemapper: an ordered list of key-rewrite rules that maps the parameter
// names found in an upstream checkpoint onto the names our modules register
// (and vice-versa). This is how we honor "don't change the models, change the
// code to adapt to it": each model variant ships a small remapper describing how
// its upstream's state_dict keys differ from ours, so a checkpoint loads 1:1
// without renaming anything in the upstream weights or in our module tree.
//
// Rules are applied in registration order. A key that any Drop rule matches is
// removed (e.g. BatchNorm "num_batches_tracked" buffers we don't keep).

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "detr/weights/state_dict.hpp"

namespace detr::weights {

class WeightRemapper {
 public:
  // Rule holds a compiled std::regex, so the special members are defined in the
  // .cpp (where Rule is complete) to keep <regex> out of this header.
  WeightRemapper();
  WeightRemapper(const WeightRemapper&);
  WeightRemapper(WeightRemapper&&);
  WeightRemapper& operator=(const WeightRemapper&);
  WeightRemapper& operator=(WeightRemapper&&);
  ~WeightRemapper();

  // Removes |prefix| from the start of a key if present (e.g. "model.").
  WeightRemapper& StripPrefix(std::string prefix);

  // Prepends |prefix| to every (surviving) key.
  WeightRemapper& AddPrefix(std::string prefix);

  // Exact-match rename of one key.
  WeightRemapper& Rename(std::string from, std::string to);

  // std::regex search/replace (ECMAScript syntax) applied to the whole key.
  WeightRemapper& ReplaceRegex(std::string pattern, std::string replacement);

  // Drops any key matching the std::regex |pattern|.
  WeightRemapper& Drop(std::string pattern);

  // Maps one key. Returns nullopt if the key is dropped.
  std::optional<std::string> Map(std::string_view key) const;

  // Applies Map to every tensor in |src|, preserving tensor data, insertion
  // order (of surviving keys), and metadata.
  StateDict Apply(const StateDict& src) const;

 private:
  struct Rule;
  std::vector<Rule> rules_;
};

}  // namespace detr::weights
