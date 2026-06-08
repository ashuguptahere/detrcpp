# Parity golden generator — torch side. Only when LibTorch is enabled.
if(DETR_ENABLE_TORCH)
  add_executable(detr-golden main.cpp)
  target_link_libraries(detr-golden PRIVATE detr::train)
  set_target_properties(detr-golden PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
endif()
