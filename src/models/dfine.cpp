// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/dfine.hpp"

#include <utility>

#include "detr/models/registry.hpp"

namespace detr::models {

DFineImpl::DFineImpl(DFineConfig cfg) : cfg_(std::move(cfg)) {
  std::vector<int> in_ch;
  if (cfg_.dinov3_sta) {
    dino_backbone_ = register_module(
        "backbone", DinoV3Sta(cfg_.vit_embed_dim, cfg_.vit_num_heads, cfg_.vit_depth, cfg_.vit_patch,
                              cfg_.interaction_indexes, cfg_.sta_inplane, cfg_.hidden_dim));
    in_ch = dino_backbone_->out_channels();
  } else {
    backbone_ = register_module("backbone", HgNetV2(cfg_.backbone, cfg_.use_lab, cfg_.return_idx));
    in_ch = backbone_->out_channels();
  }
  if (cfg_.lite_encoder) {
    lite_encoder_ = register_module(
        "encoder", DfLiteEncoder(in_ch[0], cfg_.hidden_dim, cfg_.expansion, cfg_.depth_mult));
  } else {
    encoder_ = register_module(
        "encoder", DfHybridEncoder(in_ch, cfg_.feat_strides, cfg_.hidden_dim, cfg_.nhead, cfg_.neck_ffn,
                                   cfg_.expansion, cfg_.depth_mult, cfg_.use_encoder_idx,
                                   cfg_.num_encoder_layers, 10000.0, cfg_.decoder_deimv2,
                                   cfg_.neck_repelan5));
  }
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
  dc.silu = cfg_.decoder_silu || cfg_.decoder_deimv2;  // DEIMv2 MLPs use SiLU too
  dc.deimv2 = cfg_.decoder_deimv2;
  dc.use_gateway = cfg_.decoder_gateway;
  decoder_ = register_module("decoder", DFINETransformer(dc));
}

Detections DFineImpl::Forward(torch::Tensor images) {
  auto feats = dino_backbone_ ? dino_backbone_->forward(images) : backbone_->forward(images);
  auto neck_outs = lite_encoder_ ? lite_encoder_->forward(feats) : encoder_->forward(feats);
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
  m.focal = true;                       // sigmoid head, no no-object slot
  m.imagenet_norm = cfg_.imagenet_norm;  // DINOv3-STA normalizes; HGNetv2 sizes use raw [0,1]
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
    // DEIM (Intellindust-AI-Lab/DEIM): the same D-FINE graph trained with the DEIM
    // recipe — the one architectural change is the decoder's SiLU activation (D-FINE
    // uses ReLU). Loads DEIM-D-FINE checkpoints; higher mAP than the base D-FINE.
    DFineConfig dc = SizeConfig(sz);
    dc.name = "deim-" + sz;
    dc.upstream = "https://github.com/Intellindust-AI-Lab/DEIM";
    dc.decoder_silu = true;
    RegisterOne(dc);
  }
  // DEIMv2 (Intellindust-AI-Lab/DEIMv2): the HGNetv2-backbone N variant reuses D-FINE-N's
  // backbone + neck with the new DEIMv2 decoder (RMSNorm + SwiGLU, no enc_output).
  // (atto/femto/pico need a lite encoder + micro HGNetv2; s/m/l/x need ViT/DINOv3 — TBD.)
  {
    DFineConfig v2 = SizeConfig("n");
    v2.name = "deimv2-n";
    v2.upstream = "https://github.com/Intellindust-AI-Lab/DEIMv2";
    v2.decoder_deimv2 = true;
    RegisterOne(v2);
  }
  // DEIMv2 micro sizes: a 3-stage HGNetv2 + the 2-scale LiteEncoder + the DEIMv2 decoder.
  // Each evaluates at its own native resolution (the published mAP is measured there).
  struct Micro {
    const char* name;
    const char* backbone;
    int hidden, dec_ffn, imgsz;
  };
  for (const Micro& m : {Micro{"atto", "Atto", 64, 160, 320}, Micro{"femto", "Femto", 96, 256, 416},
                         Micro{"pico", "Pico", 112, 320, 640}}) {
    DFineConfig v2;
    v2.name = std::string("deimv2-") + m.name;
    v2.upstream = "https://github.com/Intellindust-AI-Lab/DEIMv2";
    v2.imgsz = m.imgsz;
    v2.backbone = m.backbone;
    v2.use_lab = true;
    v2.return_idx = {2};
    v2.hidden_dim = m.hidden;
    v2.dec_hidden_dim = m.hidden;
    v2.feat_strides = {16, 32};
    v2.num_levels = 2;
    v2.num_points = {4, 2};
    v2.num_layers = 3;
    v2.dec_ffn = m.dec_ffn;
    v2.expansion = 0.34;
    v2.depth_mult = 0.5;
    v2.lite_encoder = true;
    v2.decoder_deimv2 = true;
    v2.decoder_gateway = false;  // micro models use a plain norm2, not the gate
    RegisterOne(v2);
  }
  // DEIMv2 s/m: a DINOv3-STA backbone (distilled RoPE ViT-Tiny + Spatial-Tuning Adapter)
  // feeding the DEIMv2 HybridEncoder neck (sum fusion + CSPLayer2) and decoder. 3 levels,
  // ImageNet normalization, 640x640. (l/x use the larger Meta DINOv3 ViT — a follow-up.)
  struct DinoSize {
    const char* name;
    int embed_dim, num_heads, hidden, neck_ffn;
    double expansion, depth_mult;
  };
  for (const DinoSize& d : {DinoSize{"s", 192, 3, 192, 512, 0.34, 0.67},
                            DinoSize{"m", 256, 4, 256, 512, 0.67, 1.0}}) {
    DFineConfig v2;
    v2.name = std::string("deimv2-") + d.name;
    v2.upstream = "https://github.com/Intellindust-AI-Lab/DEIMv2";
    v2.imgsz = 640;
    v2.imagenet_norm = true;
    v2.dinov3_sta = true;
    v2.vit_embed_dim = d.embed_dim;
    v2.vit_num_heads = d.num_heads;
    v2.interaction_indexes = {3, 7, 11};
    v2.sta_inplane = 16;
    v2.hidden_dim = d.hidden;
    v2.dec_hidden_dim = d.hidden;
    v2.feat_strides = {8, 16, 32};
    v2.num_levels = 3;
    v2.neck_ffn = d.neck_ffn;
    v2.expansion = d.expansion;
    v2.depth_mult = d.depth_mult;
    v2.use_encoder_idx = {2};
    v2.num_points = {3, 6, 3};
    v2.num_layers = 4;
    v2.dec_ffn = 512;
    v2.decoder_deimv2 = true;  // DEIMv2 neck (sum fusion) + decoder, gateway on
    v2.neck_repelan5 = true;   // version=deim RepNCSPELAN5 fuse block
    RegisterOne(v2);
  }
}

}  // namespace detr::models
