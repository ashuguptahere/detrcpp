// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Parity helper (torch side). Builds a DETR from a YAML config, runs a fixed
// seeded input through the eager LibTorch model, and writes the weights, the
// input, and the reference outputs as .safetensors so the torch-free ONNX parity
// tool can export the same weights and compare. Built with DETR_ENABLE_TORCH.

#include <cstdio>
#include <filesystem>
#include <string>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "detr/models/detr.hpp"
#include "detr/models/registry.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/torch_bridge.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: detr-golden <config.yaml> <outdir>\n");
    return 2;
  }
  const std::string config = argv[1];
  const std::string out = argv[2];

  detr::models::RegisterBuiltins();
  YAML::Node cfg = YAML::LoadFile(config);
  auto built = detr::models::Registry::Instance().Build("detr", cfg);
  if (!built) {
    std::fprintf(stderr, "build: %s\n", built.error().message.c_str());
    return 1;
  }
  auto model = *built;
  model->eval();

  torch::manual_seed(1234);
  const int imgsz = model->Meta().imgsz;
  auto input = torch::randn({1, 3, imgsz, imgsz});

  torch::NoGradGuard no_grad;
  auto outputs = model->Forward(input);

  std::error_code ec;
  std::filesystem::create_directories(out, ec);

  auto weights = detr::weights::StateDictFromModule(*model);
  if (auto r = detr::weights::SaveSafetensors(out + "/weights.safetensors", weights); !r) {
    std::fprintf(stderr, "%s\n", r.error().message.c_str());
    return 1;
  }

  detr::weights::StateDict in_sd;
  in_sd.Set("input", *detr::weights::FromTensor(input));
  detr::weights::SaveSafetensors(out + "/input.safetensors", in_sd);

  detr::weights::StateDict golden;
  golden.Set("logits", *detr::weights::FromTensor(outputs.logits));
  golden.Set("boxes", *detr::weights::FromTensor(outputs.boxes));
  detr::weights::SaveSafetensors(out + "/golden.safetensors", golden);

  std::printf("golden written to %s (imgsz=%d, %zu weights)\n", out.c_str(), imgsz,
              weights.Size());
  return 0;
}
