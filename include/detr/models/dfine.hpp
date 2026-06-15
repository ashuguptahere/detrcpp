// Copyright 2026 detrcpp authors. Apache-2.0.
//
// D-FINE (Peterande/D-FINE): an HGNetv2 backbone -> HybridEncoder neck -> the FDR
// deformable decoder. A real-time DETR (RT-DETR fork) whose novelty is the decoder's
// Fine-grained Distribution Refinement box head. Like RT-DETR: contiguous 80-class
// sigmoid head, raw-[0,1] square-resized input (no ImageNet norm). Sizes n/s/m/l/x.
// Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <vector>

#include "detr/models/dfine_decoder.hpp"
#include "detr/models/dfine_encoder.hpp"
#include "detr/models/hgnetv2.hpp"
#include "detr/models/model.hpp"

namespace detr::models {

// Configures one member of the D-FINE size matrix. Defaults are D-FINE-L.
struct DFineConfig {
  std::string name = "dfine-l";
  std::string upstream = "https://github.com/Peterande/D-FINE";
  int num_classes = 80, num_queries = 300, imgsz = 640;
  // Backbone (HGNetv2 variant + LearnableAffineBlock + returned levels).
  std::string backbone = "B4";
  bool use_lab = false;
  std::vector<int> return_idx = {1, 2, 3};
  // Backbone/neck width and the decoder's (possibly narrower) internal width — they
  // differ only for D-FINE-X (neck 384, decoder 256), which then projects in the decoder.
  int hidden_dim = 256;
  int dec_hidden_dim = 256;
  std::vector<int> feat_strides = {8, 16, 32};
  int num_levels = 3;
  // Neck (HybridEncoder, or DEIMv2 LiteEncoder for the micro sizes).
  int nhead = 8, neck_ffn = 1024, num_encoder_layers = 1;
  double expansion = 1.0, depth_mult = 1.0;
  std::vector<int> use_encoder_idx = {2};
  bool lite_encoder = false;  // DEIMv2 atto/femto/pico: 1 feature -> 2-scale LiteEncoder
  // Decoder (FDR).
  std::vector<int> num_points = {3, 6, 3};
  int num_layers = 6, dec_ffn = 1024, reg_max = 32;
  double reg_scale = 4.0;
  bool decoder_silu = false;    // D-FINE uses ReLU; DEIM (training recipe) uses SiLU
  bool decoder_deimv2 = false;  // DEIMv2 decoder: RMSNorm + SwiGLU, no enc_output, SiLU MLPs
  bool decoder_gateway = true;  // gated fusion vs plain norm2 (DEIMv2 micro = false)
};

class DFineImpl : public IModel {
 public:
  explicit DFineImpl(DFineConfig cfg);
  Detections Forward(torch::Tensor images) override;
  ModelMeta Meta() const override;

 private:
  DFineConfig cfg_;
  HgNetV2 backbone_{nullptr};
  DfHybridEncoder encoder_{nullptr};       // standard neck
  DfLiteEncoder lite_encoder_{nullptr};    // DEIMv2 micro neck (exactly one is built)
  DFINETransformer decoder_{nullptr};
};

// Registers dfine-{n,s,m,l,x} (COCO) and dfine-{s,m,l,x}-obj (Objects365->COCO) into
// the global Registry. The architecture is identical per size; the -obj entries differ
// only in provenance (the weights you load).
void RegisterDFine();

}  // namespace detr::models
