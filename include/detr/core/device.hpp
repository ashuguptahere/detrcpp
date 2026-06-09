// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Device — typed identifier for the inference / training target. Parsed from
// strings like "cuda:0", "mps", "axelera:1", "cpu", or "auto". Used by every
// backend in infer/ and by the trainer in train/.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "detr/core/result.hpp"

namespace detr::core {

enum class DeviceKind : std::uint8_t {
  Auto,
  Cpu,
  Cuda,
  Mps,
  Vulkan,
  CoreMl,
  Nnapi,
  Hailo,
  Axelera,
  MemryX,
  DeepX,
  Jetson,
};

struct Device {
  DeviceKind kind{DeviceKind::Auto};
  int index{0};        // e.g. cuda:0 → 0
  std::string serial;  // optional vendor-specific identifier
};

std::string_view ToString(DeviceKind k);
std::string ToString(const Device& d);

// Parses one device spec ("cuda:0", "auto", "axelera:1", ...).
Result<Device> ParseDevice(std::string_view spec);

// Parses a comma-separated list ("cuda:0,1,2,3" or "cuda:0,cpu") into a vector
// of Devices. A bare numeric (e.g. "1" after "cuda:0,") inherits the previous
// kind, matching PyTorch's DDP convention.
Result<std::vector<Device>> ParseDeviceList(std::string_view spec);

}  // namespace detr::core
