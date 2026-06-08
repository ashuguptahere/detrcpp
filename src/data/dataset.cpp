// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/data/dataset.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string_view>

#include <fmt/format.h>

#include "detr/data/coco.hpp"
#include "detr/data/sample.hpp"
#include "detr/data/yolo.hpp"

namespace detr::data {

namespace fs = std::filesystem;

std::string_view ToString(Split s) {
  switch (s) {
    case Split::Train: return "train";
    case Split::Val:   return "val";
    case Split::Test:  return "test";
  }
  return "train";
}

std::string_view ToString(Format f) {
  switch (f) {
    case Format::Auto: return "auto";
    case Format::Coco: return "coco";
    case Format::Yolo: return "yolo";
    case Format::Detr: return "detr";
  }
  return "auto";
}

std::size_t Dataset::CountOf(Split s) const {
  return static_cast<std::size_t>(
      std::count_if(samples.begin(), samples.end(),
                    [s](const Sample& x) { return x.split == s; }));
}

std::vector<std::size_t> Dataset::IndicesOf(Split s) const {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (samples[i].split == s) {
      out.push_back(i);
    }
  }
  return out;
}

void Dataset::Shuffle(std::uint64_t seed) {
  // Canonicalize first so the result depends only on the sample set + seed,
  // never on incoming order.
  std::stable_sort(samples.begin(), samples.end(),
                   [](const Sample& a, const Sample& b) {
                     return a.image_path < b.image_path;
                   });
  // Manual Fisher-Yates with mt19937_64 (a standardized engine producing the
  // same sequence on every platform) so the shuffle is bit-for-bit reproducible
  // across compilers and OSes — std::shuffle's index distribution is not.
  std::mt19937_64 rng(seed);
  for (std::size_t i = samples.size(); i > 1; --i) {
    const std::uint64_t j = rng() % i;
    std::swap(samples[i - 1], samples[static_cast<std::size_t>(j)]);
  }
}

Format DetectFormat(const fs::path& root) {
  std::error_code ec;
  if (fs::is_regular_file(root, ec)) {
    const std::string ext = root.extension().string();
    if (ext == ".parquet" || ext == ".detr") {
      return Format::Detr;
    }
    if (ext == ".json") {
      return Format::Coco;
    }
    if (ext == ".yaml" || ext == ".yml") {
      return Format::Yolo;
    }
    return Format::Auto;
  }
  if (fs::is_directory(root, ec)) {
    if (fs::exists(root / "data.yaml", ec) || fs::exists(root / "data.yml", ec)) {
      return Format::Yolo;
    }
    if (fs::exists(root / "annotations", ec)) {
      return Format::Coco;
    }
    for (const auto& e : fs::directory_iterator(root, ec)) {
      if (e.path().extension() == ".parquet") {
        return Format::Detr;
      }
    }
  }
  return Format::Auto;
}

core::Result<Dataset> LoadDataset(const fs::path& root, Format format) {
  if (format == Format::Auto) {
    format = DetectFormat(root);
  }
  switch (format) {
    case Format::Coco: {
      // Standard COCO layout: <root>/annotations/instances_{train,val}2017.json
      // with images under <root>/{train,val}2017. Falls back to treating |root|
      // as a single annotation json.
      std::error_code ec;
      if (fs::is_regular_file(root, ec)) {
        return LoadCocoJson(root, root.parent_path(), Split::Val);
      }
      Dataset merged;
      bool any = false;
      const std::pair<const char*, Split> splits[] = {
          {"instances_train2017.json", Split::Train},
          {"instances_val2017.json", Split::Val},
      };
      for (const auto& [name, split] : splits) {
        const fs::path ann = root / "annotations" / name;
        if (!fs::exists(ann, ec)) {
          continue;
        }
        const std::string stem =
            split == Split::Train ? "train2017" : "val2017";
        auto part = LoadCocoJson(ann, root / stem, split);
        if (!part) {
          return part;
        }
        if (!any) {
          merged.class_names = part->class_names;
          any = true;
        }
        merged.samples.insert(merged.samples.end(), part->samples.begin(),
                              part->samples.end());
      }
      if (!any) {
        return core::Err(core::ErrorCode::NotFound,
                         fmt::format("no COCO annotations under '{}'", root.string()));
      }
      return merged;
    }
    case Format::Yolo:
      return LoadYolo(root);
    case Format::Detr:
      return core::Err(core::ErrorCode::Unsupported,
                       "native 'detr' Parquet format requires building with "
                       "-DDETR_ENABLE_ARROW=ON (Apache Arrow)");
    case Format::Auto:
      return core::Err(core::ErrorCode::InvalidArgument,
                       fmt::format("could not detect dataset format at '{}'",
                                   root.string()));
  }
  return core::Err(core::ErrorCode::Internal, "unreachable");
}

}  // namespace detr::data
