// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/remapper.hpp"

#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <utility>

namespace detr::weights {

struct WeightRemapper::Rule {
  enum class Kind { StripPrefix, AddPrefix, Rename, Replace, Drop };
  Kind kind;
  std::string a;       // prefix / from / pattern
  std::string b;       // to / replacement
  std::regex re;       // compiled for Replace / Drop
  bool has_re{false};
};

// Special members defined here, where Rule is complete.
WeightRemapper::WeightRemapper() = default;
WeightRemapper::WeightRemapper(const WeightRemapper&) = default;
WeightRemapper::WeightRemapper(WeightRemapper&&) = default;
WeightRemapper& WeightRemapper::operator=(const WeightRemapper&) = default;
WeightRemapper& WeightRemapper::operator=(WeightRemapper&&) = default;
WeightRemapper::~WeightRemapper() = default;

WeightRemapper& WeightRemapper::StripPrefix(std::string prefix) {
  rules_.push_back(Rule{Rule::Kind::StripPrefix, std::move(prefix), {}, {}, false});
  return *this;
}

WeightRemapper& WeightRemapper::AddPrefix(std::string prefix) {
  rules_.push_back(Rule{Rule::Kind::AddPrefix, std::move(prefix), {}, {}, false});
  return *this;
}

WeightRemapper& WeightRemapper::Rename(std::string from, std::string to) {
  rules_.push_back(Rule{Rule::Kind::Rename, std::move(from), std::move(to), {}, false});
  return *this;
}

WeightRemapper& WeightRemapper::ReplaceRegex(std::string pattern, std::string replacement) {
  Rule r{Rule::Kind::Replace, pattern, std::move(replacement),
         std::regex(pattern, std::regex::ECMAScript), true};
  rules_.push_back(std::move(r));
  return *this;
}

WeightRemapper& WeightRemapper::Drop(std::string pattern) {
  Rule r{Rule::Kind::Drop, pattern, {}, std::regex(pattern, std::regex::ECMAScript), true};
  rules_.push_back(std::move(r));
  return *this;
}

std::optional<std::string> WeightRemapper::Map(std::string_view key) const {
  std::string k(key);
  for (const auto& r : rules_) {
    switch (r.kind) {
      case Rule::Kind::StripPrefix:
        if (k.rfind(r.a, 0) == 0) {  // starts_with
          k.erase(0, r.a.size());
        }
        break;
      case Rule::Kind::AddPrefix:
        k.insert(0, r.a);
        break;
      case Rule::Kind::Rename:
        if (k == r.a) {
          k = r.b;
        }
        break;
      case Rule::Kind::Replace:
        k = std::regex_replace(k, r.re, r.b);
        break;
      case Rule::Kind::Drop:
        if (std::regex_search(k, r.re)) {
          return std::nullopt;
        }
        break;
    }
  }
  return k;
}

StateDict WeightRemapper::Apply(const StateDict& src) const {
  StateDict out;
  for (const auto& key : src.Keys()) {
    auto mapped = Map(key);
    if (!mapped) {
      continue;
    }
    out.Set(std::move(*mapped), *src.Find(key));
  }
  for (const auto& [k, v] : src.Meta()) {
    out.SetMeta(k, v);
  }
  return out;
}

}  // namespace detr::weights
