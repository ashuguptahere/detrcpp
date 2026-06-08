add_executable(detrcpp main.cpp)

target_link_libraries(detrcpp
  PRIVATE
    detr::core
    CLI11::CLI11
    spdlog::spdlog
    fmt::fmt
    detr_warnings
)

set_target_properties(detrcpp PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

# When built with LibTorch, the CLI can enumerate models and actually train.
if(DETR_ENABLE_TORCH)
  target_link_libraries(detrcpp PRIVATE detr::train)
endif()

install(TARGETS detrcpp RUNTIME DESTINATION bin)
