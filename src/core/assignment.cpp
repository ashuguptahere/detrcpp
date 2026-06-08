// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/core/assignment.hpp"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace detr::core {

namespace {

// Classic O(n^3) Hungarian (potentials method) for a rectangular cost matrix
// with rows <= cols. cost is row-major [n x m], n <= m. Returns, for each row,
// the column it is assigned to. Adapted from the standard assignment-problem
// formulation; minimizes total cost.
std::vector<int> HungarianRowsLeqCols(const std::vector<double>& cost, int n, int m) {
  constexpr double kInf = std::numeric_limits<double>::max() / 4;
  // 1-indexed working arrays, following the canonical formulation.
  std::vector<double> u(static_cast<std::size_t>(n) + 1, 0.0);
  std::vector<double> v(static_cast<std::size_t>(m) + 1, 0.0);
  std::vector<int> p(static_cast<std::size_t>(m) + 1, 0);  // p[j] = row matched to col j
  std::vector<int> way(static_cast<std::size_t>(m) + 1, 0);

  auto sz = [](int x) { return static_cast<std::size_t>(x); };
  auto at = [&](int i, int j) {  // i in [1,n], j in [1,m], cost is 0-indexed
    return cost[sz(i - 1) * sz(m) + sz(j - 1)];
  };

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(sz(m) + 1, kInf);
    std::vector<char> used(sz(m) + 1, 0);
    do {
      used[sz(j0)] = 1;
      const int i0 = p[sz(j0)];
      double delta = kInf;
      int j1 = -1;
      for (int j = 1; j <= m; ++j) {
        if (used[sz(j)]) {
          continue;
        }
        const double cur = at(i0, j) - u[sz(i0)] - v[sz(j)];
        if (cur < minv[sz(j)]) {
          minv[sz(j)] = cur;
          way[sz(j)] = j0;
        }
        if (minv[sz(j)] < delta) {
          delta = minv[sz(j)];
          j1 = j;
        }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[sz(j)]) {
          u[sz(p[sz(j)])] += delta;
          v[sz(j)] -= delta;
        } else {
          minv[sz(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[sz(j0)] != 0);
    do {
      const int j1 = way[sz(j0)];
      p[sz(j0)] = p[sz(j1)];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> row_to_col(sz(n), -1);
  for (int j = 1; j <= m; ++j) {
    if (p[sz(j)] >= 1 && p[sz(j)] <= n) {
      row_to_col[sz(p[sz(j)] - 1)] = j - 1;
    }
  }
  return row_to_col;
}

}  // namespace

std::vector<std::pair<int, int>> LinearSumAssignment(const std::vector<double>& cost,
                                                     int n_rows, int n_cols) {
  std::vector<std::pair<int, int>> result;
  if (n_rows <= 0 || n_cols <= 0) {
    return result;
  }

  if (n_rows <= n_cols) {
    auto row_to_col = HungarianRowsLeqCols(cost, n_rows, n_cols);
    for (int r = 0; r < n_rows; ++r) {
      if (row_to_col[static_cast<std::size_t>(r)] >= 0) {
        result.emplace_back(r, row_to_col[static_cast<std::size_t>(r)]);
      }
    }
  } else {
    // Transpose so rows <= cols, solve, then swap the pair back.
    std::vector<double> t(static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(n_cols));
    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < n_cols; ++c) {
        t[static_cast<std::size_t>(c) * static_cast<std::size_t>(n_rows) +
          static_cast<std::size_t>(r)] =
            cost[static_cast<std::size_t>(r) * static_cast<std::size_t>(n_cols) +
                 static_cast<std::size_t>(c)];
      }
    }
    auto col_to_row = HungarianRowsLeqCols(t, n_cols, n_rows);
    for (int c = 0; c < n_cols; ++c) {
      if (col_to_row[static_cast<std::size_t>(c)] >= 0) {
        result.emplace_back(col_to_row[static_cast<std::size_t>(c)], c);
      }
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace detr::core
