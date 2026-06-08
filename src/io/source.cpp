// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/io/source.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <regex>
#include <string>
#include <vector>

#include <fmt/format.h>

namespace detr::io {

namespace fs = std::filesystem;
using core::Err;
using core::ErrorCode;
using core::Result;

namespace {

bool IsImageExt(const fs::path& p) {
  std::string e = p.extension().string();
  std::transform(e.begin(), e.end(), e.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp" || e == ".webp" ||
         e == ".ppm" || e == ".pgm" || e == ".tga" || e == ".gif";
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

// Translates a shell glob (* and ?) into an anchored ECMAScript regex.
std::regex GlobToRegex(const std::string& glob) {
  std::string re = "^";
  for (const char c : glob) {
    switch (c) {
      case '*': re += ".*"; break;
      case '?': re += '.'; break;
      case '.': case '(': case ')': case '+': case '|': case '^':
      case '$': case '\\': case '{': case '}': case '[': case ']':
        re += '\\';
        re += c;
        break;
      default: re += c;
    }
  }
  re += '$';
  return std::regex(re);
}

}  // namespace

Result<std::vector<std::string>> ResolveImageSources(const std::string& spec) {
  if (spec.empty()) {
    return Err(ErrorCode::InvalidArgument, "empty source");
  }
  if (StartsWith(spec, "http://") || StartsWith(spec, "https://") ||
      StartsWith(spec, "rtsp://") || StartsWith(spec, "webcam:") || spec == "-") {
    return Err(ErrorCode::Unsupported,
               fmt::format("source '{}': URLs / video / webcam land in Phase 3 "
                           "(libcurl + FFmpeg); use an image file, directory, or glob",
                           spec));
  }

  std::vector<std::string> out;
  std::error_code ec;

  if (spec.find('*') != std::string::npos || spec.find('?') != std::string::npos) {
    const fs::path pattern(spec);
    const fs::path dir = pattern.has_parent_path() ? pattern.parent_path() : fs::path(".");
    const std::regex re = GlobToRegex(pattern.filename().string());
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
      if (entry.is_regular_file(ec) &&
          std::regex_match(entry.path().filename().string(), re) && IsImageExt(entry.path())) {
        out.push_back(entry.path().string());
      }
    }
  } else if (fs::is_directory(spec, ec)) {
    for (const auto& entry : fs::recursive_directory_iterator(spec, ec)) {
      if (entry.is_regular_file(ec) && IsImageExt(entry.path())) {
        out.push_back(entry.path().string());
      }
    }
  } else if (fs::is_regular_file(spec, ec)) {
    out.push_back(spec);
  } else {
    return Err(ErrorCode::NotFound, fmt::format("source '{}' not found", spec));
  }

  if (out.empty()) {
    return Err(ErrorCode::NotFound, fmt::format("no images matched '{}'", spec));
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace detr::io
