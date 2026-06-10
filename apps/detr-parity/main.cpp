// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity gate (ONNX side, torch-free). Reads the golden produced by detr-golden:
// exports the same weights to ONNX with ExportDetr, runs the ONNX in
// onnxruntime on the same input, and compares to the reference outputs. Exits 0
// iff the max absolute difference is within tolerance. Built with
// DETR_ENABLE_ONNX.

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "detr/onnxexport/detr_export.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/tensor.hpp"

namespace {

int GetInt(const YAML::Node& c, const char* key, int def) {
  return (c && c[key]) ? c[key].as<int>() : def;
}

detr::onnxexport::DetrArch ArchFromYaml(const YAML::Node& c) {
  detr::onnxexport::DetrArch a;
  const std::string model = (c && c["model"]) ? c["model"].as<std::string>() : "detr";
  a.backbone = (model == "detr-r50" || model == "detr-r101" || model == "conditional-detr" ||
                model == "dab-detr" || model == "deformable-detr" || model == "dino")
                   ? detr::onnxexport::Backbone::ResNet50
                   : detr::onnxexport::Backbone::Compact;
  if (model == "detr-r101") {
    a.resnet_blocks = {3, 4, 23, 3};
  }
  a.hidden_dim = GetInt(c, "hidden_dim", a.hidden_dim);
  a.nheads = GetInt(c, "nheads", a.nheads);
  a.enc_layers = GetInt(c, "enc_layers", a.enc_layers);
  a.dec_layers = GetInt(c, "dec_layers", a.dec_layers);
  a.dim_feedforward = GetInt(c, "dim_feedforward", a.dim_feedforward);
  a.num_queries = GetInt(c, "num_queries", a.num_queries);
  a.num_classes = GetInt(c, "num_classes", a.num_classes);
  a.imgsz = GetInt(c, "imgsz", a.imgsz);
  a.backbone_width = GetInt(c, "backbone_width", a.backbone_width);
  a.num_levels = GetInt(c, "num_levels", a.num_levels);
  a.num_points = GetInt(c, "num_points", a.num_points);
  return a;
}

std::vector<float> AsFloats(const detr::weights::RawTensor& t) {
  std::vector<float> v(t.data.size() / sizeof(float));
  std::memcpy(v.data(), t.data.data(), t.data.size());
  return v;
}

double MaxAbsDiff(const float* a, const std::vector<float>& b, std::size_t n) {
  double m = 0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: detr-parity <config.yaml> <golden-dir> [tol]\n");
    return 2;
  }
  const std::string config = argv[1];
  const std::string dir = argv[2];
  const double tol = argc > 3 ? std::stod(argv[3]) : 1e-3;

  const YAML::Node cfg = YAML::LoadFile(config);
  const auto arch = ArchFromYaml(cfg);
  const std::string model = (cfg && cfg["model"]) ? cfg["model"].as<std::string>() : "detr";

  auto weights = detr::weights::LoadSafetensors(dir + "/weights.safetensors");
  if (!weights) {
    std::fprintf(stderr, "weights: %s\n", weights.error().message.c_str());
    return 1;
  }
  const std::string onnx_path = dir + "/model.onnx";
  auto export_r =
      (model == "conditional-detr") ? detr::onnxexport::ExportConditional(arch, *weights, onnx_path)
      : (model == "dab-detr")       ? detr::onnxexport::ExportDab(arch, *weights, onnx_path)
      : (model == "deformable-detr") ? detr::onnxexport::ExportDeformable(arch, *weights, onnx_path)
      : (model == "dino")            ? detr::onnxexport::ExportDino(arch, *weights, onnx_path)
                                     : detr::onnxexport::ExportDetr(arch, *weights, onnx_path);
  if (!export_r) {
    std::fprintf(stderr, "export: %s\n", export_r.error().message.c_str());
    return 1;
  }

  auto in_sd = detr::weights::LoadSafetensors(dir + "/input.safetensors");
  auto gold = detr::weights::LoadSafetensors(dir + "/golden.safetensors");
  if (!in_sd || !gold) {
    std::fprintf(stderr, "missing input/golden safetensors\n");
    return 1;
  }
  const auto* inp = in_sd->Find("input");
  const auto* g_logits = gold->Find("logits");
  const auto* g_boxes = gold->Find("boxes");
  if (inp == nullptr || g_logits == nullptr || g_boxes == nullptr) {
    std::fprintf(stderr, "input/golden tensors not found\n");
    return 1;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "detr-parity");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(1);
  // Run the graph exactly as emitted (the purest parity check) and avoid an ORT
  // graph-fusion bug that duplicates node names on Slice-heavy graphs (DAB's
  // dynamic 4D sine).
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  Ort::Session session(env, onnx_path.c_str(), so);
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<std::int64_t> in_shape = inp->shape;
  std::vector<float> in_data = AsFloats(*inp);
  Ort::Value in_val = Ort::Value::CreateTensor<float>(mem, in_data.data(), in_data.size(),
                                                      in_shape.data(), in_shape.size());

  const char* in_names[] = {"images"};
  const char* out_names[] = {"logits", "boxes"};
  auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in_val, 1, out_names, 2);

  const float* ort_logits = outs[0].GetTensorMutableData<float>();
  const float* ort_boxes = outs[1].GetTensorMutableData<float>();
  const std::size_t n_logits = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
  const std::size_t n_boxes = outs[1].GetTensorTypeAndShapeInfo().GetElementCount();

  const auto golden_logits = AsFloats(*g_logits);
  const auto golden_boxes = AsFloats(*g_boxes);
  if (n_logits != golden_logits.size() || n_boxes != golden_boxes.size()) {
    std::fprintf(stderr, "shape mismatch: logits %zu vs %zu, boxes %zu vs %zu\n", n_logits,
                 golden_logits.size(), n_boxes, golden_boxes.size());
    return 1;
  }

  const double d_logits = MaxAbsDiff(ort_logits, golden_logits, n_logits);
  const double d_boxes = MaxAbsDiff(ort_boxes, golden_boxes, n_boxes);
  std::printf("parity: max|Δlogits|=%.3e  max|Δboxes|=%.3e  (tol=%.1e)\n", d_logits, d_boxes, tol);

  if (d_logits <= tol && d_boxes <= tol) {
    std::printf("PARITY OK — ONNX matches LibTorch\n");
    return 0;
  }
  std::fprintf(stderr, "PARITY FAILED\n");
  return 1;
}
