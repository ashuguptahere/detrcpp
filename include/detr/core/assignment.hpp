// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Linear sum assignment (the Hungarian / Kuhn–Munkres algorithm). Shared utility
// — DETR's set-prediction matcher and the multi-object trackers both reduce to
// "match predictions to targets at minimum total cost", so it lives in core
// (DRY). O(n^3), exact, handles rectangular cost matrices.

#pragma once

#include <utility>
#include <vector>

namespace detr::core {

// Solves min-cost assignment for a row-major |cost| matrix of shape
// [n_rows x n_cols]. Returns min(n_rows, n_cols) matched (row, col) pairs whose
// total cost is minimal; every index on the smaller side is matched exactly
// once. Returns empty if either dimension is 0.
std::vector<std::pair<int, int>> LinearSumAssignment(const std::vector<double>& cost,
                                                     int n_rows, int n_cols);

}  // namespace detr::core
