// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/remapper.hpp"

#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <utility>

namespace detr::weights {

struct WeightRemapper::Rule {
  enum class Kind { StripPrefix, AddPrefix, Rename, Replace, Drop, SplitRows, SliceRows };
  Kind kind{};
  std::string a{};  // prefix / from / pattern
  std::string b{};  // to / replacement
  std::regex re{};  // compiled for Replace / Drop / Split/SliceRows
  bool has_re{false};
  std::vector<std::string> parts{};  // SplitRows: per-output replacement strings
  std::int64_t count{0};             // SliceRows: number of leading rows to keep
};

// Special members defined here, where Rule is complete.
WeightRemapper::WeightRemapper() = default;
WeightRemapper::WeightRemapper(const WeightRemapper&) = default;
WeightRemapper::WeightRemapper(WeightRemapper&&) = default;
WeightRemapper& WeightRemapper::operator=(const WeightRemapper&) = default;
WeightRemapper& WeightRemapper::operator=(WeightRemapper&&) = default;
WeightRemapper::~WeightRemapper() = default;

WeightRemapper& WeightRemapper::StripPrefix(std::string prefix) {
  rules_.push_back(Rule{.kind = Rule::Kind::StripPrefix, .a = std::move(prefix)});
  return *this;
}

WeightRemapper& WeightRemapper::AddPrefix(std::string prefix) {
  rules_.push_back(Rule{.kind = Rule::Kind::AddPrefix, .a = std::move(prefix)});
  return *this;
}

WeightRemapper& WeightRemapper::Rename(std::string from, std::string to) {
  rules_.push_back(Rule{.kind = Rule::Kind::Rename, .a = std::move(from), .b = std::move(to)});
  return *this;
}

WeightRemapper& WeightRemapper::ReplaceRegex(std::string pattern, std::string replacement) {
  rules_.push_back(Rule{.kind = Rule::Kind::Replace,
                        .a = pattern,
                        .b = std::move(replacement),
                        .re = std::regex(pattern, std::regex::ECMAScript),
                        .has_re = true});
  return *this;
}

WeightRemapper& WeightRemapper::Drop(std::string pattern) {
  rules_.push_back(Rule{.kind = Rule::Kind::Drop,
                        .a = pattern,
                        .re = std::regex(pattern, std::regex::ECMAScript),
                        .has_re = true});
  return *this;
}

WeightRemapper& WeightRemapper::SplitRows(std::string pattern,
                                          std::vector<std::string> replacements) {
  rules_.push_back(Rule{.kind = Rule::Kind::SplitRows,
                        .a = pattern,
                        .re = std::regex(pattern, std::regex::ECMAScript),
                        .has_re = true,
                        .parts = std::move(replacements)});
  return *this;
}

WeightRemapper& WeightRemapper::SliceRows(std::string pattern, std::int64_t count) {
  rules_.push_back(Rule{.kind = Rule::Kind::SliceRows,
                        .a = pattern,
                        .re = std::regex(pattern, std::regex::ECMAScript),
                        .has_re = true,
                        .count = count});
  return *this;
}

std::optional<std::string> WeightRemapper::Map(std::string_view key) const {
  std::string k(key);
  for (const auto& r : rules_) {
    switch (r.kind) {
      case Rule::Kind::StripPrefix:
        if (k.starts_with(r.a)) {
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
      case Rule::Kind::SplitRows:
      case Rule::Kind::SliceRows:
        break;  // tensor-reshaping; handled in Apply() against the mapped key
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
    const RawTensor* t = src.Find(key);
    // A SplitRows rule matching the mapped key emits N row-slices instead of one tensor.
    bool split = false;
    for (const auto& r : rules_) {
      if (r.kind != Rule::Kind::SplitRows || !std::regex_search(*mapped, r.re)) {
        continue;
      }
      const auto n = static_cast<std::int64_t>(r.parts.size());
      if (n == 0 || t->shape.empty() || t->shape[0] % n != 0) {
        break;  // not divisible into n row-blocks: leave as a single tensor
      }
      const std::size_t bytes_per = t->data.size() / static_cast<std::size_t>(n);
      for (std::size_t i = 0; i < r.parts.size(); ++i) {
        RawTensor part;
        part.dtype = t->dtype;
        part.shape = t->shape;
        part.shape[0] = t->shape[0] / n;
        const auto off = static_cast<std::ptrdiff_t>(i * bytes_per);
        part.data.assign(t->data.begin() + off,
                         t->data.begin() + off + static_cast<std::ptrdiff_t>(bytes_per));
        out.Set(std::regex_replace(*mapped, r.re, r.parts[i]), std::move(part));
      }
      split = true;
      break;
    }
    if (split) {
      continue;
    }
    // A SliceRows rule keeps only the first |count| rows of the matched tensor.
    RawTensor kept = *t;
    for (const auto& r : rules_) {
      if (r.kind != Rule::Kind::SliceRows || !std::regex_search(*mapped, r.re)) {
        continue;
      }
      if (kept.shape.empty() || r.count <= 0 || r.count >= kept.shape[0]) {
        break;  // nothing to trim
      }
      const std::size_t row_bytes = kept.data.size() / static_cast<std::size_t>(kept.shape[0]);
      kept.data.resize(static_cast<std::size_t>(r.count) * row_bytes);
      kept.shape[0] = r.count;
      break;
    }
    out.Set(std::move(*mapped), std::move(kept));
  }
  for (const auto& [k, v] : src.Meta()) {
    out.SetMeta(k, v);
  }
  return out;
}

}  // namespace detr::weights
