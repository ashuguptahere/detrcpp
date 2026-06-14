cmake_minimum_required(VERSION 3.24)

# Single source of truth for the project version: the top-level VERSION file.
# Edit it (or run scripts/bump_version.py) — never hard-code a version anywhere.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" DETR_VERSION_RAW LIMIT_COUNT 1)
string(STRIP "${DETR_VERSION_RAW}" DETR_VERSION)
if(NOT DETR_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "detrcpp: VERSION file must contain MAJOR.MINOR.PATCH, got '${DETR_VERSION}'")
endif()

project(detrcpp
  VERSION ${DETR_VERSION}
  DESCRIPTION "C++20 DETR-family detection framework"
  LANGUAGES CXX
)

set(DETR_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(DETR_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(DETR_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# Generate the canonical version header consumed via #include "detr/version.hpp".
set(DETR_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated/include")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/detr/version.hpp.in"
  "${DETR_GENERATED_INCLUDE_DIR}/detr/version.hpp"
  @ONLY
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Project-wide options.
option(DETR_BUILD_TESTS     "Build unit tests"                ON)
option(DETR_BUILD_BENCHMARKS "Build microbenchmarks"          OFF)
option(DETR_BUILD_GUI       "Build Clay+SDL3 GUI (Phase 6)"   OFF)
option(DETR_ENABLE_CUDA     "Enable CUDA / TensorRT path"     OFF)
option(DETR_ENABLE_ONNX     "Build the hand-written ONNX exporter (onnx+protobuf)" OFF)
option(DETR_ASAN            "AddressSanitizer (Debug only)"   OFF)
option(DETR_TSAN            "ThreadSanitizer (Debug only)"    OFF)
option(DETR_USE_CCACHE      "Wrap compiler with ccache/sccache" ON)

include(cpu_throttle)
include(license_scan)

# Enable CUDA as a first-class language only when requested AND available, so
# the release-cuda preset's CMAKE_CUDA_ARCHITECTURES is actually honored.
if(DETR_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "detrcpp: CUDA enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(WARNING "detrcpp: DETR_ENABLE_CUDA=ON but no CUDA compiler found; "
                    "disabling the CUDA/TensorRT path.")
    set(DETR_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA / TensorRT path" FORCE)
  endif()
endif()

# ccache / sccache wrapper — speeds up rebuilds significantly.
if(DETR_USE_CCACHE)
  find_program(SCCACHE_BIN sccache)
  find_program(CCACHE_BIN  ccache)
  if(SCCACHE_BIN)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_BIN}")
    message(STATUS "detrcpp: using sccache at ${SCCACHE_BIN}")
  elseif(CCACHE_BIN)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_BIN}")
    message(STATUS "detrcpp: using ccache at ${CCACHE_BIN}")
  endif()
endif()

# Warnings — Google-style, treat the dangerous ones as errors.
add_library(detr_warnings INTERFACE)
if(MSVC)
  target_compile_options(detr_warnings INTERFACE /W4 /permissive- /Zc:__cplusplus)
else()
  target_compile_options(detr_warnings INTERFACE
    -Wall -Wextra -Wpedantic
    -Wshadow -Wnon-virtual-dtor -Wold-style-cast
    -Wcast-align -Wunused -Woverloaded-virtual
    -Wconversion -Wsign-conversion -Wnull-dereference
    -Wdouble-promotion -Wformat=2
    -Werror=return-type -Werror=switch
  )
endif()

# Sanitizers — opt-in, debug builds only.
if((DETR_ASAN OR DETR_TSAN) AND NOT MSVC)
  if(DETR_ASAN)
    target_compile_options(detr_warnings INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(detr_warnings INTERFACE -fsanitize=address)
  endif()
  if(DETR_TSAN)
    target_compile_options(detr_warnings INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(detr_warnings INTERFACE -fsanitize=thread)
  endif()
endif()

# Dependencies — fetched from source and built via CMake FetchContent (pinned
# tags), so a plain `cmake` configure needs no package manager. See
# cmake/dependencies.cmake. LibTorch (below) is the sole find_package SDK.
include(dependencies)

# LibTorch — only when the training/inference path is requested. Point CMake at
# it with -DCMAKE_PREFIX_PATH=/path/to/libtorch (or -DTorch_DIR=...).
if(DETR_ENABLE_TORCH)
  find_package(Torch REQUIRED)
  message(STATUS "detrcpp: LibTorch found (${TORCH_INSTALL_PREFIX})")
endif()

if(DETR_ENABLE_ONNX)
  # The ONNX exporter is kept torch-free (config + safetensors -> graph) and
  # built without LibTorch, so the FetchContent protobuf and LibTorch's bundled
  # protobuf never meet in one link (protobuf + onnx are fetched in
  # cmake/dependencies.cmake when DETR_ENABLE_ONNX is on).
  if(DETR_ENABLE_TORCH)
    message(FATAL_ERROR
      "DETR_ENABLE_ONNX and DETR_ENABLE_TORCH cannot be enabled together: their "
      "protobuf providers conflict. Build the ONNX exporter as a separate "
      "torch-off configuration.")
  endif()
  message(STATUS "detrcpp: ONNX exporter enabled")
  # onnxruntime (prebuilt) for the parity gate — optional; set -DONNXRUNTIME_ROOT.
  find_path(DETR_ORT_INCLUDE onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_ROOT}/include)
  find_library(DETR_ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
  if(DETR_ORT_INCLUDE AND DETR_ORT_LIB)
    set(DETR_ORT_FOUND TRUE)
    message(STATUS "detrcpp: onnxruntime found (${DETR_ORT_LIB})")
  else()
    message(STATUS "detrcpp: onnxruntime not found; set -DONNXRUNTIME_ROOT to enable the parity tool")
  endif()
endif()

# Subdirectories.
add_subdirectory(src)
add_subdirectory(apps/detrcpp)
add_subdirectory(apps/detrcpp-export)
add_subdirectory(apps/detr-golden)
add_subdirectory(apps/detr-parity)

if(DETR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(DETR_BUILD_BENCHMARKS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
  else()
    message(WARNING "detrcpp: DETR_BUILD_BENCHMARKS=ON but benchmarks/CMakeLists.txt "
                    "is missing; skipping benchmarks.")
  endif()
endif()

# License-scan: print a summary and fail if any disallowed dep slipped in.
detr_license_scan()

message(STATUS "")
message(STATUS "detrcpp ${PROJECT_VERSION} configured.")
message(STATUS "  C++:        ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Generator:  ${CMAKE_GENERATOR}")
message(STATUS "  Tests:      ${DETR_BUILD_TESTS}")
message(STATUS "  GUI:        ${DETR_BUILD_GUI}")
message(STATUS "  CUDA/TRT:   ${DETR_ENABLE_CUDA}")
message(STATUS "")
