# Parity checker — ONNX side (torch-free, uses onnxruntime). Only when the ONNX
# exporter and onnxruntime are both available.
if(DETR_ENABLE_ONNX AND DETR_ORT_FOUND)
  add_executable(detr-parity main.cpp)
  target_link_libraries(detr-parity
    PRIVATE detr::onnx detr_weights yaml-cpp::yaml-cpp "${DETR_ORT_LIB}")
  target_include_directories(detr-parity SYSTEM PRIVATE ${DETR_ORT_INCLUDE})
  set_target_properties(detr-parity PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
endif()
