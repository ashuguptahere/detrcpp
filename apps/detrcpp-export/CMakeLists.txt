# detrcpp-export — torch-free ONNX exporter binary. Only when the ONNX exporter
# is enabled (DETR_ENABLE_TORCH must be off in this configuration).
if(DETR_ENABLE_ONNX)
  add_executable(detrcpp-export main.cpp)
  target_link_libraries(detrcpp-export
    PRIVATE detr::onnx detr_weights CLI11::CLI11 yaml-cpp::yaml-cpp detr_warnings)
  set_target_properties(detrcpp-export PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  install(TARGETS detrcpp-export RUNTIME DESTINATION bin)
endif()
