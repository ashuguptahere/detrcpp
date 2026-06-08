// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/onnxexport/graph_builder.hpp"

#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>

namespace detr::onnxexport {

namespace {

void SetTensorShape(onnx::TypeProto* type, const std::vector<std::int64_t>& dims) {
  auto* tt = type->mutable_tensor_type();
  tt->set_elem_type(onnx::TensorProto::FLOAT);
  auto* shape = tt->mutable_shape();
  for (const std::int64_t d : dims) {
    auto* dim = shape->add_dim();
    if (d < 0) {
      dim->set_dim_param("N");  // dynamic
    } else {
      dim->set_dim_value(d);
    }
  }
}

}  // namespace

struct GraphBuilder::Impl {
  onnx::GraphProto graph;
  onnx::NodeProto* last_node{nullptr};
  std::unordered_map<std::string, int> counters;
};

GraphBuilder::GraphBuilder(std::string graph_name) : impl_(std::make_unique<Impl>()) {
  impl_->graph.set_name(std::move(graph_name));
}

GraphBuilder::~GraphBuilder() = default;
GraphBuilder::GraphBuilder(GraphBuilder&&) noexcept = default;
GraphBuilder& GraphBuilder::operator=(GraphBuilder&&) noexcept = default;

void GraphBuilder::AddInput(const std::string& name, const std::vector<std::int64_t>& dims) {
  auto* vi = impl_->graph.add_input();
  vi->set_name(name);
  SetTensorShape(vi->mutable_type(), dims);
}

void GraphBuilder::AddOutput(const std::string& name, const std::vector<std::int64_t>& dims) {
  auto* vi = impl_->graph.add_output();
  vi->set_name(name);
  SetTensorShape(vi->mutable_type(), dims);
}

void GraphBuilder::AddInitializerF32(const std::string& name,
                                     const std::vector<std::int64_t>& dims,
                                     const std::vector<float>& data) {
  auto* t = impl_->graph.add_initializer();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::FLOAT);
  for (const std::int64_t d : dims) {
    t->add_dims(d);
  }
  t->mutable_float_data()->Add(data.begin(), data.end());
}

void GraphBuilder::AddInitializerI64(const std::string& name,
                                     const std::vector<std::int64_t>& dims,
                                     const std::vector<std::int64_t>& data) {
  auto* t = impl_->graph.add_initializer();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::INT64);
  for (const std::int64_t d : dims) {
    t->add_dims(d);
  }
  t->mutable_int64_data()->Add(data.begin(), data.end());
}

void GraphBuilder::AddNode(const std::string& op_type, const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs, const std::string& name) {
  auto* n = impl_->graph.add_node();
  n->set_op_type(op_type);
  if (!name.empty()) {
    n->set_name(name);
  }
  for (const auto& in : inputs) {
    n->add_input(in);
  }
  for (const auto& out : outputs) {
    n->add_output(out);
  }
  impl_->last_node = n;
}

void GraphBuilder::AttrInt(const std::string& name, std::int64_t value) {
  auto* a = impl_->last_node->add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::INT);
  a->set_i(value);
}

void GraphBuilder::AttrInts(const std::string& name, const std::vector<std::int64_t>& values) {
  auto* a = impl_->last_node->add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::INTS);
  for (const std::int64_t v : values) {
    a->add_ints(v);
  }
}

void GraphBuilder::AttrFloat(const std::string& name, float value) {
  auto* a = impl_->last_node->add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::FLOAT);
  a->set_f(value);
}

void GraphBuilder::AttrString(const std::string& name, const std::string& value) {
  auto* a = impl_->last_node->add_attribute();
  a->set_name(name);
  a->set_type(onnx::AttributeProto::STRING);
  a->set_s(value);
}

std::string GraphBuilder::Unique(const std::string& prefix) {
  const int n = impl_->counters[prefix]++;
  return fmt::format("{}_{}", prefix, n);
}

core::Result<void> GraphBuilder::Save(const std::string& path, int opset_version) {
  onnx::ModelProto model;
  model.set_ir_version(onnx::IR_VERSION);
  model.set_producer_name("detrcpp");
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(opset_version);
  *model.mutable_graph() = impl_->graph;

  try {
    onnx::checker::check_model(model);
  } catch (const std::exception& e) {
    return core::Err(core::ErrorCode::Internal,
                     fmt::format("ONNX model failed validation: {}", e.what()));
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return core::Err(core::ErrorCode::Io, fmt::format("cannot write '{}'", path));
  }
  if (!model.SerializeToOstream(&out)) {
    return core::Err(core::ErrorCode::Io, fmt::format("failed to serialize '{}'", path));
  }
  return {};
}

}  // namespace detr::onnxexport
