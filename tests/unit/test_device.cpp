// Copyright 2026 detrcpp authors. Apache-2.0.

#include <gtest/gtest.h>

#include <string>

#include "detr/core/device.hpp"

namespace detr::core {

TEST(ParseDevice, Simple) {
  auto r = ParseDevice("cpu");
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->kind, DeviceKind::Cpu);
  EXPECT_EQ(r->index, 0);
}

TEST(ParseDevice, CudaWithIndex) {
  auto r = ParseDevice("cuda:3");
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_EQ(r->kind, DeviceKind::Cuda);
  EXPECT_EQ(r->index, 3);
  EXPECT_EQ(ToString(*r), "cuda:3");
}

TEST(ParseDevice, AutoNoIndex) {
  auto r = ParseDevice("auto");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->kind, DeviceKind::Auto);
  EXPECT_EQ(ToString(*r), "auto");
}

TEST(ParseDevice, VendorSerial) {
  auto r = ParseDevice("axelera:abc123");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->kind, DeviceKind::Axelera);
  EXPECT_EQ(r->serial, "abc123");
}

TEST(ParseDevice, CaseInsensitive) {
  auto r = ParseDevice("CUDA:0");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->kind, DeviceKind::Cuda);
}

TEST(ParseDevice, RejectsUnknown) {
  auto r = ParseDevice("toaster:0");
  EXPECT_FALSE(r.has_value());
  EXPECT_EQ(r.error().code, ErrorCode::InvalidArgument);
}

TEST(ParseDevice, RejectsEmpty) {
  auto r = ParseDevice("");
  EXPECT_FALSE(r.has_value());
}

TEST(ParseDeviceList, MultipleCuda) {
  auto r = ParseDeviceList("cuda:0,cuda:1,cuda:2");
  ASSERT_TRUE(r.has_value()) << r.error().message;
  ASSERT_EQ(r->size(), 3U);
  EXPECT_EQ((*r)[0].index, 0);
  EXPECT_EQ((*r)[2].index, 2);
}

TEST(ParseDeviceList, BareNumericInheritsKind) {
  auto r = ParseDeviceList("cuda:0,1,2,3");
  ASSERT_TRUE(r.has_value()) << r.error().message;
  ASSERT_EQ(r->size(), 4U);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ((*r)[i].kind, DeviceKind::Cuda);
    EXPECT_EQ((*r)[i].index, static_cast<int>(i));
  }
}

TEST(ParseDeviceList, MixedKinds) {
  auto r = ParseDeviceList("cpu,cuda:0,mps");
  ASSERT_TRUE(r.has_value());
  ASSERT_EQ(r->size(), 3U);
  EXPECT_EQ((*r)[0].kind, DeviceKind::Cpu);
  EXPECT_EQ((*r)[1].kind, DeviceKind::Cuda);
  EXPECT_EQ((*r)[2].kind, DeviceKind::Mps);
}

TEST(ParseDeviceList, BareWithoutPriorKindFails) {
  auto r = ParseDeviceList("1,2");
  EXPECT_FALSE(r.has_value());
}

}  // namespace detr::core
