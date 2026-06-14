# dependencies.cmake — third-party libraries pulled from source via CMake
# FetchContent, pinned to exact upstream tags. No package manager: a plain
# `cmake` configure fetches and builds these (cached under build/_deps).
#
# Pins match the versions detrcpp was last validated against. LibTorch is the
# one exception — it is a large prebuilt SDK, still located with find_package(Torch)
# and pointed at via -DCMAKE_PREFIX_PATH (see the top-level CMakeLists).
#
# Each dep's SPDX license is asserted in cmake/license_scan.cmake (run after this
# module) — keep the two in sync when changing a pin.

include(FetchContent)

# Build deps quietly and don't let them add themselves to our install/test sets.
set(FETCHCONTENT_QUIET ON)

# ---- fmt (must come before spdlog: spdlog uses the fmt::fmt target directly) ----
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 12.1.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(fmt)

# ---- spdlog (external fmt; recent spdlog reuses the fmt::fmt target if present) ----
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.17.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(spdlog)

# ---- CLI11 (header-only) ----
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.6.2
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(cli11)

# ---- tl-expected (header-only; target tl::expected) ----
set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EXPECTED_BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(expected
  GIT_REPOSITORY https://github.com/TartanLlama/expected.git
  GIT_TAG v1.1.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(expected)

# ---- simdjson ----
set(SIMDJSON_DEVELOPER_MODE OFF CACHE BOOL "" FORCE)
set(SIMDJSON_ENABLE_THREADS ON CACHE BOOL "" FORCE)
FetchContent_Declare(simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
  GIT_TAG v4.6.4
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(simdjson)

# ---- yaml-cpp (latest release tag is 0.8.0; exports yaml-cpp::yaml-cpp) ----
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(yaml-cpp)

# ---- GoogleTest (only when tests are built) ----
if(DETR_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

# ---- Google Benchmark (only when microbenchmarks are built) ----
if(DETR_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(benchmark)
endif()

# ---- protobuf + ONNX (only for the torch-free ONNX exporter) ----
# protobuf 3.21.12 is the last release before the abseil dependency, so the
# source build stays self-contained. ONNX links it as ONNX::onnx / ONNX::onnx_proto.
if(DETR_ENABLE_ONNX)
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
  set(protobuf_ABSL_PROVIDER "module" CACHE STRING "" FORCE)
  FetchContent_Declare(protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v3.21.12
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(protobuf)

  set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ONNX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(ONNX_ML ON CACHE BOOL "" FORCE)
  FetchContent_Declare(onnx
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    GIT_TAG v1.16.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(onnx)
endif()
