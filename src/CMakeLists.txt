add_library(detr_core STATIC
  log/log.cpp
  core/device.cpp
  core/assignment.cpp
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

# detr_data — dataset adapters (COCO via simdjson, YOLO via yaml-cpp) producing
# the format-independent Dataset. LibTorch-independent; the loader/decoder that
# turns Samples into batched tensors lands with the torch-gated training code.
add_library(detr_data STATIC
  data/dataset.cpp
  data/coco.cpp
  data/yolo.cpp
)

target_include_directories(detr_data
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(detr_data
  PUBLIC
    detr_core
  PRIVATE
    simdjson::simdjson
    yaml-cpp::yaml-cpp
    fmt::fmt
    detr_warnings
)

set_target_properties(detr_data PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

add_library(detr::data ALIAS detr_data)

# detr_torch — LibTorch bridge for the weight layer (RawTensor <-> torch::Tensor,
# load StateDict into a live module by name). Gated on DETR_ENABLE_TORCH. Torch
# headers are included as SYSTEM so their warnings don't trip our strict flags;
# we deliberately do not attach detr_warnings here.
if(DETR_ENABLE_TORCH)
  add_library(detr_torch STATIC
    weights/torch_bridge.cpp
  )
  target_link_libraries(detr_torch
    PUBLIC
      detr_weights
      "${TORCH_LIBRARIES}"
    PRIVATE
      fmt::fmt
  )
  target_include_directories(detr_torch SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
  target_compile_definitions(detr_torch PUBLIC DETR_ENABLE_TORCH)
  separate_arguments(_detr_torch_flags UNIX_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(detr_torch PUBLIC ${_detr_torch_flags})
  set_target_properties(detr_torch PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
  add_library(detr::torch ALIAS detr_torch)

  # detr_models — model registry + DETR variants (torch modules).
  add_library(detr_models STATIC
    models/registry.cpp
    models/detr.cpp
  )
  target_link_libraries(detr_models
    PUBLIC
      detr_weights
      "${TORCH_LIBRARIES}"
      yaml-cpp::yaml-cpp
    PRIVATE
      fmt::fmt
  )
  target_include_directories(detr_models SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
  target_compile_definitions(detr_models PUBLIC DETR_ENABLE_TORCH)
  target_compile_options(detr_models PUBLIC ${_detr_torch_flags})
  set_target_properties(detr_models PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
  add_library(detr::models ALIAS detr_models)

  # detr_train — box ops, Hungarian matcher, set criterion (DETR loss), trainer.
  add_library(detr_train STATIC
    train/box_ops.cpp
    train/matcher.cpp
    train/criterion.cpp
    train/ema.cpp
    train/checkpoint.cpp
    train/trainer.cpp
  )
  target_link_libraries(detr_train
    PUBLIC
      detr_models
      detr_torch
      detr_core
      "${TORCH_LIBRARIES}"
    PRIVATE
      fmt::fmt
  )
  target_include_directories(detr_train SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
  target_compile_definitions(detr_train PUBLIC DETR_ENABLE_TORCH)
  target_compile_options(detr_train PUBLIC ${_detr_torch_flags})
  set_target_properties(detr_train PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
  add_library(detr::train ALIAS detr_train)
endif()
