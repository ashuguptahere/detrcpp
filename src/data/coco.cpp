// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/data/coco.hpp"

#include <fmt/format.h>
#include <simdjson.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace detr::data {

namespace fs = std::filesystem;
using core::Err;
using core::ErrorCode;
using core::Result;

namespace {

bool GetInt(simdjson::dom::element e, const char* key, std::int64_t& out) {
  return e[key].get_int64().get(out) == simdjson::SUCCESS;
}

// Reads a numeric field as double, tolerating both integer and float JSON.
double GetNum(simdjson::dom::element e, const char* key) {
  double d = 0;
  if (e[key].get_double().get(d) == simdjson::SUCCESS) {
    return d;
  }
  std::int64_t i = 0;
  if (e[key].get_int64().get(i) == simdjson::SUCCESS) {
    return static_cast<double>(i);
  }
  return 0.0;
}

// Joins an untrusted COCO `file_name` under |images_dir|, rejecting absolute
// paths and any "../" traversal that would escape the dataset directory.
Result<std::string> SafeImagePath(const fs::path& images_dir, std::string_view file_name) {
  const fs::path rel{file_name};
  if (rel.is_absolute()) {
    return Err(ErrorCode::InvalidArgument,
               fmt::format("COCO file_name '{}' must be relative", file_name));
  }
  const fs::path base = images_dir.lexically_normal();
  const fs::path joined = (base / rel).lexically_normal();
  const fs::path within = joined.lexically_relative(base);
  if (within.empty() || *within.begin() == "..") {
    return Err(ErrorCode::InvalidArgument,
               fmt::format("COCO file_name '{}' escapes images_dir", file_name));
  }
  return joined.string();
}

}  // namespace

Result<Dataset> LoadCocoJson(const fs::path& annotations_json, const fs::path& images_dir,
                             Split split, bool raw_category_ids) {
  simdjson::dom::parser parser;
  simdjson::dom::element doc;
  auto err = parser.load(annotations_json.string()).get(doc);
  if (err) {
    return Err(ErrorCode::ParseError,
               fmt::format("cannot parse COCO json '{}': {}", annotations_json.string(),
                           simdjson::error_message(err)));
  }

  // 1) categories -> 0-based class index sorted by category id.
  simdjson::dom::array cats;
  if (doc["categories"].get_array().get(cats)) {
    return Err(ErrorCode::ParseError, "COCO json missing 'categories' array");
  }
  std::vector<std::pair<std::int64_t, std::string>> cat_list;
  for (auto c : cats) {
    std::int64_t id = 0;
    std::string_view name;
    if (!GetInt(c, "id", id) || c["name"].get_string().get(name)) {
      return Err(ErrorCode::ParseError, "COCO category missing id/name");
    }
    cat_list.emplace_back(id, std::string(name));
  }
  std::sort(cat_list.begin(), cat_list.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  Dataset ds;
  std::unordered_map<std::int64_t, int> cat_to_class;
  ds.class_names.reserve(cat_list.size());
  for (int i = 0; i < static_cast<int>(cat_list.size()); ++i) {
    cat_to_class[cat_list[static_cast<std::size_t>(i)].first] = i;
    ds.class_names.push_back(cat_list[static_cast<std::size_t>(i)].second);
  }

  // 2) images -> samples, remembering image_id -> sample index and dims.
  simdjson::dom::array images;
  if (doc["images"].get_array().get(images)) {
    return Err(ErrorCode::ParseError, "COCO json missing 'images' array");
  }
  std::unordered_map<std::int64_t, std::size_t> id_to_index;
  for (auto im : images) {
    std::int64_t id = 0;
    std::int64_t w = 0;
    std::int64_t h = 0;
    std::string_view file;
    if (!GetInt(im, "id", id) || im["file_name"].get_string().get(file)) {
      return Err(ErrorCode::ParseError, "COCO image missing id/file_name");
    }
    GetInt(im, "width", w);
    GetInt(im, "height", h);
    auto safe_path = SafeImagePath(images_dir, file);
    if (!safe_path) {
      return tl::make_unexpected(safe_path.error());
    }
    Sample s;
    s.image_path = std::move(*safe_path);
    s.width = static_cast<int>(w);
    s.height = static_cast<int>(h);
    s.split = split;
    id_to_index[id] = ds.samples.size();
    ds.samples.push_back(std::move(s));
  }

  // 3) annotations -> normalized cxcywh boxes appended to their image.
  simdjson::dom::array anns;
  if (doc["annotations"].get_array().get(anns)) {
    // Some COCO test splits ship images with no annotations — tolerate it.
    return ds;
  }
  for (auto a : anns) {
    std::int64_t image_id = 0;
    std::int64_t cat_id = 0;
    if (!GetInt(a, "image_id", image_id) || !GetInt(a, "category_id", cat_id)) {
      continue;
    }
    std::int64_t iscrowd = 0;
    GetInt(a, "iscrowd", iscrowd);
    // Crowd annotations are kept: they are excluded from training targets (see
    // the loader) but used as ignore regions in eval, matching pycocotools.
    simdjson::dom::array box;
    if (a["bbox"].get_array().get(box)) {
      continue;
    }
    double xywh[4] = {0, 0, 0, 0};
    int bi = 0;
    for (auto v : box) {
      if (bi < 4) {
        double d = 0;
        if (v.get_double().get(d) == simdjson::SUCCESS) {
          xywh[bi] = d;
        }
      }
      ++bi;
    }
    auto it = id_to_index.find(image_id);
    auto cls = cat_to_class.find(cat_id);
    if (it == id_to_index.end() || cls == cat_to_class.end()) {
      continue;
    }
    Sample& s = ds.samples[it->second];
    if (s.width <= 0 || s.height <= 0) {
      continue;  // cannot normalize without image dims.
    }
    const auto fw = static_cast<float>(s.width);
    const auto fh = static_cast<float>(s.height);
    BBox b;
    b.cx = (static_cast<float>(xywh[0]) + static_cast<float>(xywh[2]) / 2.0F) / fw;
    b.cy = (static_cast<float>(xywh[1]) + static_cast<float>(xywh[3]) / 2.0F) / fh;
    b.w = static_cast<float>(xywh[2]) / fw;
    b.h = static_cast<float>(xywh[3]) / fh;
    b.class_id = raw_category_ids ? static_cast<int>(cat_id) : cls->second;
    b.iscrowd = (iscrowd == 1);
    const double area_px = GetNum(a, "area");
    b.area = area_px > 0.0 ? static_cast<float>(area_px / (static_cast<double>(fw) * fh)) : 0.0F;
    s.boxes.push_back(b);
  }

  return ds;
}

}  // namespace detr::data
