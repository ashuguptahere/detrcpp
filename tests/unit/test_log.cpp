// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/log/log.hpp"

#include <gtest/gtest.h>

#include <string>

namespace detr::log {

TEST(Log, GetReturnsSameLoggerByName) {
  auto& a = Get("train.optimizer");
  auto& b = Get("train.optimizer");
  EXPECT_EQ(&a, &b);
}

TEST(Log, DifferentNamesYieldDifferentLoggers) {
  auto& a = Get("infer.trt");
  auto& b = Get("infer.onnx");
  EXPECT_NE(&a, &b);
}

TEST(Log, SetGlobalLevelAffectsExistingLoggers) {
  auto& lg = Get("test.level");
  SetGlobalLevel(Level::Debug);
  EXPECT_TRUE(lg.should_log(spdlog::level::debug));
  SetGlobalLevel(Level::Warn);
  EXPECT_FALSE(lg.should_log(spdlog::level::info));
  SetGlobalLevel(Level::Info);  // restore
}

}  // namespace detr::log
