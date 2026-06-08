// Copyright 2026 detrcpp authors. Apache-2.0.
//
// A thin builder over the official ONNX C++ proto for emitting a model by hand
// (no Python, no torch.onnx). Models call AddInput / AddInitializer* / AddNode /
// AddOutput to construct their graph, then Save() validates it with the ONNX
// checker and writes a .onnx file. PIMPL keeps protobuf/onnx headers out of the
// public interface. Compiled with DETR_ENABLE_ONNX.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "detr/core/result.hpp"

namespace detr::onnxexport {

class GraphBuilder {
 public:
  explicit GraphBuilder(std::string graph_name);
  ~GraphBuilder();
  GraphBuilder(GraphBuilder&&) noexcept;
  GraphBuilder& operator=(GraphBuilder&&) noexcept;
  GraphBuilder(const GraphBuilder&) = delete;
  GraphBuilder& operator=(const GraphBuilder&) = delete;

  // Graph float32 input/output with shape |dims| (use -1 for a dynamic dim).
  void AddInput(const std::string& name, const std::vector<std::int64_t>& dims);
  void AddOutput(const std::string& name, const std::vector<std::int64_t>& dims);

  // Constant tensors (weights). Data is row-major and must equal prod(dims).
  void AddInitializerF32(const std::string& name, const std::vector<std::int64_t>& dims,
                         const std::vector<float>& data);
  void AddInitializerI64(const std::string& name, const std::vector<std::int64_t>& dims,
                         const std::vector<std::int64_t>& data);

  // Appends a node; subsequent Attr* calls attach to it.
  void AddNode(const std::string& op_type, const std::vector<std::string>& inputs,
               const std::vector<std::string>& outputs, const std::string& name = "");
  void AttrInt(const std::string& name, std::int64_t value);
  void AttrInts(const std::string& name, const std::vector<std::int64_t>& values);
  void AttrFloat(const std::string& name, float value);
  void AttrString(const std::string& name, const std::string& value);

  // A name unique within this graph, e.g. Unique("conv") -> "conv_0".
  std::string Unique(const std::string& prefix);

  // Validates with onnx::checker and serializes to |path|.
  core::Result<void> Save(const std::string& path, int opset_version = 17);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace detr::onnxexport
