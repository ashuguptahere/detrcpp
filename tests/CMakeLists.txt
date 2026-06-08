find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(detr_unit_tests
  unit/test_device.cpp
  unit/test_log.cpp
  unit/test_safetensors.cpp
  unit/test_remapper.cpp
  unit/test_dataset.cpp
  unit/test_assignment.cpp
  unit/test_coco_eval.cpp
)

target_link_libraries(detr_unit_tests
  PRIVATE
    detr::core
    detr::weights
    detr::data
    detr::eval
    GTest::gtest
    GTest::gtest_main
    detr_warnings
)

set_target_properties(detr_unit_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

if(DETR_ENABLE_TORCH)
  target_sources(detr_unit_tests PRIVATE
    unit/test_torch_bridge.cpp
    unit/test_detr.cpp
    unit/test_train.cpp
    unit/test_loader.cpp
    unit/test_io.cpp
  )
  target_link_libraries(detr_unit_tests PRIVATE detr::torch detr::models detr::train)
endif()

if(DETR_ENABLE_ONNX)
  target_sources(detr_unit_tests PRIVATE unit/test_onnx_builder.cpp)
  target_link_libraries(detr_unit_tests PRIVATE detr::onnx)
endif()

gtest_discover_tests(detr_unit_tests
  PROPERTIES LABELS "unit"
)

# Smoke test: run the actual CLI binary and check exit codes.
add_test(
  NAME cli_help
  COMMAND detrcpp --help
)
set_tests_properties(cli_help PROPERTIES LABELS "smoke")

add_test(
  NAME cli_version
  COMMAND detrcpp --version
)
set_tests_properties(cli_version PROPERTIES
  LABELS "smoke"
  PASS_REGULAR_EXPRESSION "[0-9]+\\.[0-9]+\\.[0-9]+"
)

add_test(
  NAME cli_list_models
  COMMAND detrcpp --list-models
)
set_tests_properties(cli_list_models PROPERTIES LABELS "smoke")
