// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/dfine.hpp"

#include <utility>

#include "detr/models/registry.hpp"

namespace detr::models {

DFineImpl::DFineImpl(DFineConfig cfg) : cfg_(std::move(cfg)) {
  backbone_ = register_module("backbone", HgNetV2(cfg_.backbone, cfg_.use_lab, cfg_.return_idx));
  const auto in_ch = backbone_->out_channels();
  encoder_ = register_module(
      "encoder", DfHybridEncoder(in_ch, cfg_.feat_strides, cfg_.hidden_dim, cfg_.nhead, cfg_.neck_ffn,
                                 cfg_.expansion, cfg_.depth_mult, cfg_.use_encoder_idx,
                                 cfg_.num_encoder_layers, 10000.0));
  DfTransformerConfig dc;
  dc.num_classes = cfg_.num_classes;
  dc.hidden_dim = cfg_.dec_hidden_dim;                                   // decoder width
  dc.feat_channels =
      std::vector<int>(static_cast<std::size_t>(cfg_.num_levels), cfg_.hidden_dim);  // neck width
  dc.num_queries = cfg_.num_queries;
  dc.feat_strides = cfg_.feat_strides;
  dc.num_levels = cfg_.num_levels;
  dc.num_points = cfg_.num_points;
  dc.nhead = cfg_.nhead;
  dc.num_layers = cfg_.num_layers;
  dc.dim_feedforward = cfg_.dec_ffn;
  dc.reg_max = cfg_.reg_max;
  dc.reg_scale = cfg_.reg_scale;
  decoder_ = register_module("decoder", DFINETransformer(dc));
}

Detections DFineImpl::Forward(torch::Tensor images) {
  auto feats = backbone_->forward(images);
  auto neck_outs = encoder_->forward(feats);
  auto [logits, boxes] = decoder_->forward(neck_outs);
  Detections det;
  det.logits = logits;  // [B, Q, num_classes] (sigmoid/focal)
  det.boxes = boxes;    // [B, Q, 4] cxcywh in [0,1]
  return det;
}

ModelMeta DFineImpl::Meta() const {
  ModelMeta m;
  m.name = cfg_.name;
  m.imgsz = cfg_.imgsz;
  m.num_classes = cfg_.num_classes;
  m.num_queries = cfg_.num_queries;
  m.focal = true;             // sigmoid head, no no-object slot
  m.imagenet_norm = false;    // raw [0,1], square resize (RT-DETR/D-FINE recipe)
  m.license = "Apache-2.0";
  m.upstream = cfg_.upstream;
  return m;
}

namespace {

// Builds the per-size D-FINE config (the n/s/m/l/x knobs from configs/dfine/*).
DFineConfig SizeConfig(const std::string& size) {
  DFineConfig c;
  c.name = "dfine-" + size;
  if (size == "n") {
    c.backbone = "B0";
    c.use_lab = true;
    c.return_idx = {2, 3};
    c.hidden_dim = 128;
    c.dec_hidden_dim = 128;
    c.feat_strides = {16, 32};
    c.num_levels = 2;
    c.neck_ffn = 512;
    c.dec_ffn = 512;
    c.expansion = 0.34;
    c.depth_mult = 0.5;
    c.use_encoder_idx = {1};
    c.num_points = {6, 6};
    c.num_layers = 3;
  } else if (size == "s") {
    c.backbone = "B0";
    c.use_lab = true;
    c.expansion = 0.5;
    c.depth_mult = 0.34;
    c.num_layers = 3;
  } else if (size == "m") {
    c.backbone = "B2";
    c.use_lab = true;
    c.depth_mult = 0.67;
    c.num_layers = 4;
  } else if (size == "l") {
    c.backbone = "B4";  // defaults already match D-FINE-L
  } else if (size == "x") {
    c.backbone = "B5";
    c.hidden_dim = 384;       // neck width
    c.dec_hidden_dim = 256;   // decoder runs narrower; input_proj projects 384 -> 256
    c.neck_ffn = 2048;
    c.reg_scale = 8.0;
  }
  return c;
}

void RegisterOne(const DFineConfig& c) {
  DFineImpl probe(c);
  Registry::Instance().Register(c.name, probe.Meta(),
                                [c](const YAML::Node&) -> std::shared_ptr<IModel> {
                                  return std::make_shared<DFineImpl>(c);
                                });
}

}  // namespace

void RegisterDFine() {
  for (const std::string sz : {"n", "s", "m", "l", "x"}) {
    RegisterOne(SizeConfig(sz));
    // The Objects365->COCO variants (no nano) share the architecture; only the
    // weights you load differ (higher mAP). Named with a short `-obj` suffix.
    if (sz != "n") {
      DFineConfig oc = SizeConfig(sz);
      oc.name = "dfine-" + sz + "-obj";
      oc.upstream = "https://github.com/Peterande/D-FINE (Objects365->COCO)";
      RegisterOne(oc);
    }
  }
}

}  // namespace detr::models
