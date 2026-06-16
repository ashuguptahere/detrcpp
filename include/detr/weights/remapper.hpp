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

#include <cstdint>
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

  // Splits one tensor into N along dim 0 (rows). A key matching |pattern| (after the
  // rename rules) is replaced by |replacements|.size() keys — replacement[i] is the
  // regex substitution naming part i, which receives an equal contiguous slice of the
  // rows. Handles the fused-qkv / fused-`in_proj_weight` checkpoints that store q,k,v
  // (or q_proj/k_proj/v_proj) stacked in one tensor, vs our separate projections.
  WeightRemapper& SplitRows(std::string pattern, std::vector<std::string> replacements);

  // Keeps only the first |count| rows (dim 0) of a tensor whose key matches |pattern|
  // (after the rename rules). For checkpoints that store more entries than our module
  // uses — e.g. LW-DETR's group-DETR query embeddings ([1300, *], of which inference
  // uses the leading num_queries rows).
  WeightRemapper& SliceRows(std::string pattern, std::int64_t count);

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
