// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Structural test: build a complete StateDict for a tiny DETR architecture (with
// every weight at the correct shape) and export it to ONNX. A pass means the
// emission referenced every weight by the right name + shape and produced a
// graph that the onnx checker accepts. Numerical parity (vs LibTorch) is checked
// separately by the onnxruntime parity test.

#include "detr/onnxexport/detr_export.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "detr/weights/state_dict.hpp"
#include "detr/weights/tensor.hpp"

namespace detr::onnxexport {
namespace {

using weights::DType;
using weights::RawTensor;
using weights::StateDict;

RawTensor F32(std::vector<std::int64_t> shape) {
  RawTensor t;
  t.dtype = DType::F32;
  t.shape = std::move(shape);
  std::int64_t n = 1;
  for (auto d : t.shape) {
    n *= d;
  }
  t.data.resize(static_cast<std::size_t>(n) * sizeof(float));
  return t;
}

void AddBn(StateDict& sd, const std::string& p, int c) {
  sd.Set(p + ".weight", F32({c}));
  sd.Set(p + ".bias", F32({c}));
  sd.Set(p + ".running_mean", F32({c}));
  sd.Set(p + ".running_var", F32({c}));
}

void AddAttn(StateDict& sd, const std::string& p, int d) {
  sd.Set(p + ".in_proj_weight", F32({3 * d, d}));
  sd.Set(p + ".in_proj_bias", F32({3 * d}));
  sd.Set(p + ".out_proj.weight", F32({d, d}));
  sd.Set(p + ".out_proj.bias", F32({d}));
}

void AddNorm(StateDict& sd, const std::string& p, int d) {
  sd.Set(p + ".weight", F32({d}));
  sd.Set(p + ".bias", F32({d}));
}

StateDict BuildStateDict(const DetrArch& a) {
  StateDict sd;
  const int w0 = a.backbone_width;
  const int d = a.hidden_dim;
  const int ff = a.dim_feedforward;

  // backbone stem
  sd.Set("backbone.0.weight", F32({w0, 3, 7, 7}));
  AddBn(sd, "backbone.1", w0);
  // stages
  const int chans[4] = {w0, 2 * w0, 4 * w0, 8 * w0};
  const int base[4] = {4, 10, 16, 22};
  int in = w0;
  for (int s = 0; s < 4; ++s) {
    const int out = chans[s];
    sd.Set("backbone." + std::to_string(base[s]) + ".weight", F32({out, in, 3, 3}));
    AddBn(sd, "backbone." + std::to_string(base[s] + 1), out);
    sd.Set("backbone." + std::to_string(base[s] + 3) + ".weight", F32({out, out, 3, 3}));
    AddBn(sd, "backbone." + std::to_string(base[s] + 4), out);
    in = out;
  }

  sd.Set("input_proj.weight", F32({d, 8 * w0, 1, 1}));
  sd.Set("input_proj.bias", F32({d}));
  sd.Set("query_embed.weight", F32({a.num_queries, d}));

  for (int i = 0; i < a.enc_layers; ++i) {
    const std::string p = "encoder." + std::to_string(i);
    AddAttn(sd, p + ".self_attn", d);
    sd.Set(p + ".linear1.weight", F32({ff, d}));
    sd.Set(p + ".linear1.bias", F32({ff}));
    sd.Set(p + ".linear2.weight", F32({d, ff}));
    sd.Set(p + ".linear2.bias", F32({d}));
    AddNorm(sd, p + ".norm1", d);
    AddNorm(sd, p + ".norm2", d);
  }
  for (int i = 0; i < a.dec_layers; ++i) {
    const std::string p = "decoder." + std::to_string(i);
    AddAttn(sd, p + ".self_attn", d);
    AddAttn(sd, p + ".cross_attn", d);
    sd.Set(p + ".linear1.weight", F32({ff, d}));
    sd.Set(p + ".linear1.bias", F32({ff}));
    sd.Set(p + ".linear2.weight", F32({d, ff}));
    sd.Set(p + ".linear2.bias", F32({d}));
    AddNorm(sd, p + ".norm1", d);
    AddNorm(sd, p + ".norm2", d);
    AddNorm(sd, p + ".norm3", d);
  }

  AddNorm(sd, "decoder_norm", d);
  sd.Set("class_embed.weight", F32({a.num_classes + 1, d}));
  sd.Set("class_embed.bias", F32({a.num_classes + 1}));
  sd.Set("bbox_embed.0.weight", F32({d, d}));
  sd.Set("bbox_embed.0.bias", F32({d}));
  sd.Set("bbox_embed.2.weight", F32({d, d}));
  sd.Set("bbox_embed.2.bias", F32({d}));
  sd.Set("bbox_embed.4.weight", F32({4, d}));
  sd.Set("bbox_embed.4.bias", F32({4}));
  return sd;
}

TEST(DetrExport, EmitsCheckerValidGraph) {
  DetrArch a;
  a.hidden_dim = 32;
  a.nheads = 4;
  a.enc_layers = 1;
  a.dec_layers = 1;
  a.dim_feedforward = 64;
  a.num_queries = 5;
  a.num_classes = 4;
  a.imgsz = 64;
  a.backbone_width = 8;

  auto sd = BuildStateDict(a);
  const auto path = (std::filesystem::temp_directory_path() / "detr_export.onnx").string();
  auto r = ExportDetr(a, sd, path);
  ASSERT_TRUE(r.has_value()) << r.error().message;

  std::error_code ec;
  EXPECT_GT(std::filesystem::file_size(path, ec), 100U);
  std::filesystem::remove(path, ec);
}

TEST(DetrExport, ReportsMissingWeight) {
  DetrArch a;
  a.hidden_dim = 32;
  a.nheads = 4;
  a.enc_layers = 1;
  a.dec_layers = 1;
  a.dim_feedforward = 64;
  a.num_queries = 5;
  a.num_classes = 4;
  a.imgsz = 64;
  a.backbone_width = 8;
  auto sd = BuildStateDict(a);
  sd = StateDict{};  // empty -> every weight missing
  auto r = ExportDetr(a, sd, "/tmp/should_not_write.onnx");
  EXPECT_FALSE(r.has_value());
}

}  // namespace
}  // namespace detr::onnxexport
