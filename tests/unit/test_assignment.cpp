// Copyright 2026 detrcpp authors. Apache-2.0.

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "detr/core/assignment.hpp"

namespace detr::core {
namespace {

double TotalCost(const std::vector<double>& cost, int cols,
                 const std::vector<std::pair<int, int>>& m) {
  double t = 0;
  for (const auto& [r, c] : m) {
    t += cost[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
              static_cast<std::size_t>(c)];
  }
  return t;
}

double At(const std::vector<double>& cost, int i) {
  return cost[static_cast<std::size_t>(i)];
}

TEST(Assignment, SquareValidPermutationAndOptimal) {
  std::vector<double> cost = {1, 2, 3, 2, 4, 6, 3, 6, 9};
  auto m = LinearSumAssignment(cost, 3, 3);
  ASSERT_EQ(m.size(), 3U);
  // Each row and col used exactly once.
  std::vector<int> rows, cols;
  for (auto& [r, c] : m) {
    rows.push_back(r);
    cols.push_back(c);
  }
  std::sort(rows.begin(), rows.end());
  std::sort(cols.begin(), cols.end());
  EXPECT_EQ(rows, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(cols, (std::vector<int>{0, 1, 2}));
  // Optimal equals brute-force minimum.
  std::vector<int> perm = {0, 1, 2};
  double best = 1e18;
  do {
    best = std::min(best, At(cost, perm[0]) + At(cost, 3 + perm[1]) + At(cost, 6 + perm[2]));
  } while (std::next_permutation(perm.begin(), perm.end()));
  EXPECT_DOUBLE_EQ(TotalCost(cost, 3, m), best);
}

TEST(Assignment, MatchesBruteForceSmall) {
  std::vector<double> cost = {4, 1, 3, 2, 0, 5, 3, 2, 2};
  auto m = LinearSumAssignment(cost, 3, 3);
  // Brute-force minimum over all 3! permutations.
  std::vector<int> perm = {0, 1, 2};
  double best = 1e18;
  do {
    const double t = At(cost, perm[0]) + At(cost, 3 + perm[1]) + At(cost, 6 + perm[2]);
    best = std::min(best, t);
  } while (std::next_permutation(perm.begin(), perm.end()));
  EXPECT_DOUBLE_EQ(TotalCost(cost, 3, m), best);
}

TEST(Assignment, RectangularMoreRowsThanCols) {
  // 4 rows (predictions), 2 cols (targets): every col matched once, 2 matches.
  std::vector<double> cost = {9, 9, 1, 9, 9, 1, 9, 9};
  auto m = LinearSumAssignment(cost, 4, 2);
  ASSERT_EQ(m.size(), 2U);
  EXPECT_DOUBLE_EQ(TotalCost(cost, 2, m), 2.0);  // rows 1->0 and 2->1
}

TEST(Assignment, RectangularMoreColsThanRows) {
  std::vector<double> cost = {5, 1, 5, 5, 5, 5, 5, 1};
  auto m = LinearSumAssignment(cost, 2, 4);
  ASSERT_EQ(m.size(), 2U);
  EXPECT_DOUBLE_EQ(TotalCost(cost, 4, m), 2.0);
}

TEST(Assignment, EmptyInputs) {
  EXPECT_TRUE(LinearSumAssignment({}, 0, 3).empty());
  EXPECT_TRUE(LinearSumAssignment({}, 3, 0).empty());
}

}  // namespace
}  // namespace detr::core
