// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/data/yolo.hpp"

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "detr/data/sample.hpp"

namespace detr::data {

namespace fs = std::filesystem;
using core::Err;
using core::ErrorCode;
using core::Result;

namespace {

bool IsImage(const fs::path& p) {
  std::string e = p.extension().string();
  std::transform(e.begin(), e.end(), e.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp" || e == ".webp";
}

// Maps an image path to its YOLO label path: the last "images" path component
// becomes "labels", and the extension becomes ".txt".
fs::path LabelPathFor(const fs::path& image) {
  fs::path out;
  bool replaced = false;
  std::vector<fs::path> parts(image.begin(), image.end());
  for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
    if (!replaced && *it == "images") {
      *it = "labels";
      replaced = true;
    }
  }
  for (const auto& p : parts) {
    out /= p;
  }
  out.replace_extension(".txt");
  return out;
}

std::vector<BBox> ReadLabels(const fs::path& label) {
  std::vector<BBox> boxes;
  std::ifstream f(label);
  if (!f) {
    return boxes;
  }
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    int cls = 0;
    float cx = 0;
    float cy = 0;
    float w = 0;
    float h = 0;
    if (ss >> cls >> cx >> cy >> w >> h) {
      boxes.push_back(BBox{cx, cy, w, h, cls});
    }
  }
  return boxes;
}

// Parses YOLO `names:` as either a sequence or an int-keyed map, preserving
// class index order.
std::vector<std::string> ParseNames(const YAML::Node& names) {
  std::vector<std::string> out;
  if (names.IsSequence()) {
    for (const auto& n : names) {
      out.push_back(n.as<std::string>());
    }
  } else if (names.IsMap()) {
    std::map<int, std::string> ordered;
    for (const auto& kv : names) {
      ordered[kv.first.as<int>()] = kv.second.as<std::string>();
    }
    for (const auto& [k, v] : ordered) {
      out.push_back(v);
    }
  }
  return out;
}

void CollectSplit(const fs::path& location, Split split, std::vector<Sample>& out) {
  std::error_code ec;
  std::vector<fs::path> images;
  if (fs::is_directory(location, ec)) {
    for (const auto& e : fs::recursive_directory_iterator(location, ec)) {
      if (e.is_regular_file(ec) && IsImage(e.path())) {
        images.push_back(e.path());
      }
    }
  } else if (fs::is_regular_file(location, ec)) {
    // A .txt listing image paths (one per line).
    std::ifstream f(location);
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty()) {
        fs::path p(line);
        images.push_back(p.is_absolute() ? p : location.parent_path() / p);
      }
    }
  }
  std::sort(images.begin(), images.end());  // stable, reproducible order.
  for (const auto& img : images) {
    Sample s;
    s.image_path = img.string();
    s.split = split;
    s.boxes = ReadLabels(LabelPathFor(img));
    out.push_back(std::move(s));
  }
}

}  // namespace

Result<Dataset> LoadYolo(const fs::path& root) {
  std::error_code ec;
  fs::path yaml_path = root;
  if (fs::is_directory(root, ec)) {
    if (fs::exists(root / "data.yaml", ec)) {
      yaml_path = root / "data.yaml";
    } else if (fs::exists(root / "data.yml", ec)) {
      yaml_path = root / "data.yml";
    } else {
      return Err(ErrorCode::NotFound, fmt::format("no data.yaml under '{}'", root.string()));
    }
  }

  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(yaml_path.string());
  } catch (const std::exception& e) {
    return Err(ErrorCode::ParseError,
               fmt::format("cannot parse '{}': {}", yaml_path.string(), e.what()));
  }

  Dataset ds;
  if (cfg["names"]) {
    ds.class_names = ParseNames(cfg["names"]);
  }

  // Base path for split locations: explicit `path:` (relative to the yaml file)
  // or the yaml file's directory.
  fs::path base = yaml_path.parent_path();
  if (cfg["path"]) {
    fs::path p(cfg["path"].as<std::string>());
    base = p.is_absolute() ? p : (yaml_path.parent_path() / p);
  }

  const std::pair<const char*, Split> splits[] = {
      {"train", Split::Train}, {"val", Split::Val}, {"test", Split::Test}};
  bool any = false;
  for (const auto& [key, split] : splits) {
    if (!cfg[key]) {
      continue;
    }
    fs::path loc(cfg[key].as<std::string>());
    if (!loc.is_absolute()) {
      loc = base / loc;
    }
    CollectSplit(loc, split, ds.samples);
    any = true;
  }
  if (!any) {
    return Err(ErrorCode::NotFound,
               fmt::format("'{}' lists no train/val/test splits", yaml_path.string()));
  }
  return ds;
}

}  // namespace detr::data
