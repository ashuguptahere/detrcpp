find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(detr_unit_tests
  unit/test_device.cpp
  unit/test_log.cpp
)

target_link_libraries(detr_unit_tests
  PRIVATE
    detr::core
    GTest::gtest
    GTest::gtest_main
    detr_warnings
)

set_target_properties(detr_unit_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

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
