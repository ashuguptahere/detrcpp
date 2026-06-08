// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/core/device.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace detr::core {

namespace {

struct KindName {
  DeviceKind kind;
  std::string_view name;
};

constexpr std::array<KindName, 12> kKindNames{{
    {DeviceKind::Auto,    "auto"},
    {DeviceKind::Cpu,     "cpu"},
    {DeviceKind::Cuda,    "cuda"},
    {DeviceKind::Mps,     "mps"},
    {DeviceKind::Vulkan,  "vulkan"},
    {DeviceKind::CoreMl,  "coreml"},
    {DeviceKind::Nnapi,   "nnapi"},
    {DeviceKind::Hailo,   "hailo"},
    {DeviceKind::Axelera, "axelera"},
    {DeviceKind::MemryX,  "memryx"},
    {DeviceKind::DeepX,   "deepx"},
    {DeviceKind::Jetson,  "jetson"},
}};

std::string ToLower(std::string_view s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

// Kinds that carry a meaningful integer index (cuda:0, hailo:1, ...). Cpu, Mps,
// and Auto are singletons — an index on them is meaningless.
bool IsIndexBearing(DeviceKind k) {
  switch (k) {
    case DeviceKind::Cpu:
    case DeviceKind::Mps:
    case DeviceKind::Auto:
      return false;
    default:
      return true;
  }
}

bool AllDigits(std::string_view s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
  });
}

}  // namespace

std::string_view ToString(DeviceKind k) {
  for (const auto& kn : kKindNames) {
    if (kn.kind == k) {
      return kn.name;
    }
  }
  return "unknown";
}

std::string ToString(const Device& d) {
  if (!IsIndexBearing(d.kind)) {
    return std::string(ToString(d.kind));
  }
  if (!d.serial.empty()) {
    return fmt::format("{}:{}", ToString(d.kind), d.serial);
  }
  return fmt::format("{}:{}", ToString(d.kind), d.index);
}

Result<Device> ParseDevice(std::string_view spec) {
  if (spec.empty()) {
    return Err(ErrorCode::InvalidArgument, "empty device spec");
  }
  const std::string lower = ToLower(spec);
  const auto colon = lower.find(':');
  const std::string_view head = colon == std::string::npos
                                    ? std::string_view{lower}
                                    : std::string_view{lower.data(), colon};
  const std::string_view tail = colon == std::string::npos
                                    ? std::string_view{}
                                    : std::string_view{lower.data() + colon + 1,
                                                       lower.size() - colon - 1};

  DeviceKind kind = DeviceKind::Auto;
  bool matched = false;
  for (const auto& kn : kKindNames) {
    if (kn.name == head) {
      kind = kn.kind;
      matched = true;
      break;
    }
  }
  if (!matched) {
    return Err(ErrorCode::InvalidArgument,
               fmt::format("unknown device kind '{}'", head));
  }

  Device d;
  d.kind = kind;
  if (!tail.empty()) {
    if (AllDigits(tail)) {
      // All-digit tail is an index. A digit string that fails to parse is an
      // overflow — an error, never a silent fall-through to "serial".
      int idx = 0;
      auto [ptr, ec] = std::from_chars(tail.data(), tail.data() + tail.size(), idx);
      if (ec != std::errc{} || ptr != tail.data() + tail.size()) {
        return Err(ErrorCode::InvalidArgument,
                   fmt::format("device index out of range: '{}'", tail));
      }
      d.index = idx;
    } else {
      // Non-numeric tail → vendor serial (e.g. "axelera:abc123").
      d.serial = std::string(tail);
    }
  }
  return d;
}

Result<std::vector<Device>> ParseDeviceList(std::string_view spec) {
  std::vector<Device> out;
  if (spec.empty()) {
    return Err(ErrorCode::InvalidArgument, "empty device list");
  }

  DeviceKind previous = DeviceKind::Auto;
  bool have_previous = false;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= spec.size(); ++i) {
    if (i == spec.size() || spec[i] == ',') {
      std::string_view part = spec.substr(start, i - start);
      start = i + 1;
      if (part.empty()) {
        return Err(ErrorCode::InvalidArgument, "empty element in device list");
      }
      // Bare numeric ("cuda:0,1") inherits the previous kind.
      if (AllDigits(part)) {
        if (!have_previous || !IsIndexBearing(previous)) {
          return Err(ErrorCode::InvalidArgument,
                     "bare numeric device requires a preceding index-bearing kind "
                     "(e.g. cuda:0,1)");
        }
        Device d;
        d.kind = previous;
        int idx = 0;
        auto [ptr, ec] = std::from_chars(part.data(), part.data() + part.size(), idx);
        if (ec != std::errc{} || ptr != part.data() + part.size()) {
          return Err(ErrorCode::InvalidArgument,
                     fmt::format("device index out of range: '{}'", part));
        }
        d.index = idx;
        out.push_back(d);
        continue;
      }
      auto r = ParseDevice(part);
      if (!r) {
        return tl::make_unexpected(r.error());
      }
      previous = r->kind;
      have_previous = true;
      out.push_back(*r);
    }
  }
  return out;
}

}  // namespace detr::core
