// Copyright 2026 detrcpp authors. Apache-2.0.
//
// detrcpp-export — the torch-free ONNX exporter binary. Reads an architecture
// config + a .safetensors checkpoint and writes a validated .onnx (no Python, no
// LibTorch). Kept separate from the main `detrcpp` binary because vcpkg-protobuf
// (for onnx) and LibTorch's bundled protobuf cannot coexist in one link.

#include <yaml-cpp/yaml.h>

#include <CLI/CLI.hpp>
#include <cstdio>
#include <string>

#include "detr/onnxexport/detr_export.hpp"
#include "detr/weights/safetensors.hpp"

namespace {

int GetInt(const YAML::Node& c, const char* key, int def) {
  return (c && c[key]) ? c[key].as<int>() : def;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"detrcpp-export — export a DETR checkpoint to ONNX (Python-free)"};

  std::string model = "detr";
  std::string config;
  std::string weights;
  std::string out = "model.onnx";
  int imgsz = 0;
  app.add_option("-m,--model", model, "model architecture (currently: detr)");
  app.add_option("-c,--config", config, "YAML architecture config");
  app.add_option("-w,--weights", weights, "input .safetensors checkpoint")->required();
  app.add_option("-o,--out", out, "output .onnx path")->capture_default_str();
  app.add_option("--imgsz", imgsz, "override image size (fixed in the exported graph)");
  CLI11_PARSE(app, argc, argv);

  if (model != "detr" && model != "detr-r50" && model != "detr-r101") {
    std::fprintf(stderr, "unsupported model '%s' (have: detr, detr-r50, detr-r101)\n",
                 model.c_str());
    return 2;
  }

  YAML::Node cfg;
  if (!config.empty()) {
    try {
      cfg = YAML::LoadFile(config);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "config '%s': %s\n", config.c_str(), e.what());
      return 2;
    }
  }

  detr::onnxexport::DetrArch arch;
  arch.backbone = (model == "detr-r50" || model == "detr-r101")
                      ? detr::onnxexport::Backbone::ResNet50
                      : detr::onnxexport::Backbone::Compact;
  if (model == "detr-r101") {
    arch.resnet_blocks = {3, 4, 23, 3};
  }
  arch.hidden_dim = GetInt(cfg, "hidden_dim", arch.hidden_dim);
  arch.nheads = GetInt(cfg, "nheads", arch.nheads);
  arch.enc_layers = GetInt(cfg, "enc_layers", arch.enc_layers);
  arch.dec_layers = GetInt(cfg, "dec_layers", arch.dec_layers);
  arch.dim_feedforward = GetInt(cfg, "dim_feedforward", arch.dim_feedforward);
  arch.num_queries = GetInt(cfg, "num_queries", arch.num_queries);
  arch.num_classes = GetInt(cfg, "num_classes", arch.num_classes);
  arch.imgsz = GetInt(cfg, "imgsz", arch.imgsz);
  arch.backbone_width = GetInt(cfg, "backbone_width", arch.backbone_width);
  if (imgsz > 0) {
    arch.imgsz = imgsz;
  }

  auto sd = detr::weights::LoadSafetensors(weights);
  if (!sd) {
    std::fprintf(stderr, "weights '%s': %s\n", weights.c_str(), sd.error().message.c_str());
    return 1;
  }
  if (auto r = detr::onnxexport::ExportDetr(arch, *sd, out); !r) {
    std::fprintf(stderr, "export: %s\n", r.error().message.c_str());
    return 1;
  }
  std::printf("exported %s -> %s  (input images[1,3,%d,%d])\n", model.c_str(), out.c_str(),
              arch.imgsz, arch.imgsz);
  return 0;
}
