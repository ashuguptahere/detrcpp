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
