add_library(detr_core STATIC
  log/log.cpp
  core/device.cpp
)

target_include_directories(detr_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${DETR_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(detr_core
  PUBLIC
    spdlog::spdlog
    fmt::fmt
    tl::expected
    detr_warnings
)

set_target_properties(detr_core PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

add_library(detr::core ALIAS detr_core)

# detr_weights — framework-independent weight interchange (safetensors, state
# dict, key remapping). No LibTorch dependency, so it builds and tests anywhere;
# the optional torch bridge is added separately when DETR_ENABLE_TORCH is on.
add_library(detr_weights STATIC
  weights/tensor.cpp
  weights/state_dict.cpp
  weights/safetensors.cpp
  weights/remapper.cpp
)

target_include_directories(detr_weights
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(detr_weights
  PUBLIC
    detr_core
  PRIVATE
    simdjson::simdjson
    fmt::fmt
    detr_warnings
)

set_target_properties(detr_weights PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

add_library(detr::weights ALIAS detr_weights)
