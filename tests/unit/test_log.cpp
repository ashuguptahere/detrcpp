// Copyright 2026 detrcpp authors. Apache-2.0.

#include <gtest/gtest.h>

#include <string>

#include "detr/log/log.hpp"

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
  EXPECT_TRUE(lg.ShouldLog(Level::Debug));
  SetGlobalLevel(Level::Warn);
  EXPECT_FALSE(lg.ShouldLog(Level::Info));
  SetGlobalLevel(Level::Info);  // restore
}

}  // namespace detr::log
