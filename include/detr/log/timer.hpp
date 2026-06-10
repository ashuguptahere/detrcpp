// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Stopwatch: a tiny monotonic wall-clock timer for hot-path instrumentation
// (training step, eval/predict throughput, data-vs-compute split). Header-only,
// allocation-free, and steady_clock-based so it is immune to wall-clock jumps.
// Pair it with the detr::log facade to emit the measured numbers.

#pragma once

#include <chrono>

namespace detr::log {

class Stopwatch {
 public:
  Stopwatch() : start_(std::chrono::steady_clock::now()) {}

  void Reset() { start_ = std::chrono::steady_clock::now(); }

  double ElapsedMs() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_)
        .count();
  }
  double ElapsedSec() const { return ElapsedMs() / 1000.0; }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace detr::log
