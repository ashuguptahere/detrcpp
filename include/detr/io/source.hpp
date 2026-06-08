// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Resolves a --source spec into concrete image paths. Today it handles a single
// image file, a directory (recursively), and a shell-style glob (* and ?).
// URLs, video, RTSP, and webcam indices are recognized and reported as not-yet-
// supported (Phase 3 brings libcurl + FFmpeg), so the CLI gives a clear message
// rather than a confusing failure.

#pragma once

#include <string>
#include <vector>

#include "detr/core/result.hpp"

namespace detr::io {

core::Result<std::vector<std::string>> ResolveImageSources(const std::string& spec);

}  // namespace detr::io
