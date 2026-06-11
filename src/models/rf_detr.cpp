// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/rf_detr.hpp"

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include "detr/models/deform_attn.hpp"  // SpatialShapes
#include "detr/models/deform_head.hpp"
#include "detr/models/model.hpp"
#include "detr/models/registry.hpp"
#include "detr/models/vit.hpp"

namespace detr::models {

namespace {

namespace nn = torch::nn;

struct Config {
  int vit_embed{256};
  int vit_depth{4};
  int vit_heads{8};
  int patch{16};
  int hidden_dim{256};
  int nheads{8};
  int dec_layers{6};
  int dim_feedforward{1024};
  int num_queries{300};
  int num_classes{90};
  int num_levels{3};
  int num_points{4};
  int imgsz{640};
};

template <typename T>
T Get(const YAML::Node& c, const char* k, T fb) {
  return (c && c[k]) ? c[k].as<T>() : fb;
}

Config ReadConfig(const YAML::Node& c) {
  Config x;
  x.vit_embed = Get(c, "vit_embed", x.vit_embed);
  x.vit_depth = Get(c, "vit_depth", x.vit_depth);
  x.vit_heads = Get(c, "vit_heads", x.vit_heads);
  x.patch = Get(c, "patch", x.patch);
  x.hidden_dim = Get(c, "hidden_dim", x.hidden_dim);
  x.nheads = Get(c, "nheads", x.nheads);
  x.dec_layers = Get(c, "dec_layers", x.dec_layers);
  x.dim_feedforward = Get(c, "dim_feedforward", x.dim_feedforward);
  x.num_queries = Get(c, "num_queries", x.num_queries);
  x.num_classes = Get(c, "num_classes", x.num_classes);
  x.num_levels = Get(c, "num_levels", x.num_levels);
  x.num_points = Get(c, "num_points", x.num_points);
  x.imgsz = Get(c, "imgsz", x.imgsz);
  return x;
}

class RfDetrImpl : public IModel {
 public:
  explicit RfDetrImpl(Config cfg, bool denoising = false) : cfg_(cfg), denoising_(denoising) {
    const int d = cfg.hidden_dim;
    backbone_ =
        register_module("backbone", ViT(cfg.vit_embed, cfg.vit_depth, cfg.vit_heads, cfg.patch, 4));

    // Multi-scale projection from the single ViT feature: 1x1 then strided 3x3.
    input_proj_ = register_module("input_proj", nn::ModuleList());
    input_proj_->push_back(nn::Sequential(nn::Conv2d(nn::Conv2dOptions(cfg.vit_embed, d, 1)),
                                          nn::GroupNorm(nn::GroupNormOptions(32, d))));
    for (int i = 1; i < cfg.num_levels; ++i) {
      input_proj_->push_back(
          nn::Sequential(nn::Conv2d(nn::Conv2dOptions(d, d, 3).stride(2).padding(1)),
                         nn::GroupNorm(nn::GroupNormOptions(32, d))));
    }
    head_ = BuildDeformDetectHead(*this, d, cfg.num_levels, cfg.nheads, cfg.num_points,
                                  cfg.dim_feedforward, cfg.dec_layers, cfg.num_classes,
                                  cfg.num_queries);
    if (denoising_) {
      // RF-DETR-CDN: per-class content embedding for denoising queries (+1 unused
      // row). Registered only for rf-detr-cdn, so plain rf-detr stays byte-identical.
      label_enc_ = register_module("label_enc", nn::Embedding(cfg.num_classes + 1, d));
    }
  }

  Detections Forward(torch::Tensor images) override {
    auto enc = Encode(images);
    return RunDeformDetectHead(head_, enc.memory, enc.shapes);
  }

  bool SupportsDenoising() const override { return denoising_; }

  Detections ForwardDenoise(torch::Tensor images, const DenoisingInput& dn_in,
                            DenoisingOut& dn_out) override {
    if (!denoising_ || !dn_in.active || !is_training()) {
      dn_out = DenoisingOut{};
      return Forward(images);
    }
    auto enc = Encode(images);
    // The deform head is batch-major [B, L, *]; DenoisingInput is query-major.
    DeformCdn cdn;
    cdn.active = true;
    cdn.num_dn = dn_in.num_dn;
    cdn.dn_tgt = label_enc_->forward(dn_in.dn_labels.transpose(0, 1).contiguous());  // [B,num_dn,d]
    cdn.dn_ref = dn_in.dn_ref.transpose(0, 1).contiguous();                          // [B,num_dn,4]
    cdn.attn_mask = dn_in.attn_mask;
    return RunDeformDetectHead(head_, enc.memory, enc.shapes, cdn, dn_out);
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = denoising_ ? "rf-detr-cdn" : "rf-detr";
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.focal = true;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/roboflow/rf-detr";
    return m;
  }

 private:
  struct Encoded {
    torch::Tensor memory;  // [B, Sum(HW), d]
    SpatialShapes shapes;
  };

  // ViT backbone + the multi-scale projection -> the deformable head memory.
  Encoded Encode(torch::Tensor images) {
    auto feat = backbone_->forward(images);  // [B, vit_embed, h, w]
    std::vector<torch::Tensor> srcs;
    srcs.push_back(input_proj_[0]->as<nn::SequentialImpl>()->forward(feat));
    for (int i = 1; i < cfg_.num_levels; ++i) {
      srcs.push_back(
          input_proj_[static_cast<std::size_t>(i)]->as<nn::SequentialImpl>()->forward(srcs.back()));
    }
    SpatialShapes shapes;
    std::vector<torch::Tensor> mem;
    for (const auto& s : srcs) {
      shapes.emplace_back(s.size(2), s.size(3));
      mem.push_back(s.flatten(2).transpose(1, 2));
    }
    return {torch::cat(mem, 1), shapes};
  }

  Config cfg_;
  bool denoising_{false};
  ViT backbone_{nullptr};
  nn::ModuleList input_proj_{nullptr};
  DeformDetectHead head_;
  nn::Embedding label_enc_{nullptr};
};

}  // namespace

std::shared_ptr<IModel> MakeRfDetr(const YAML::Node& cfg) {
  return std::make_shared<RfDetrImpl>(ReadConfig(cfg));
}

ModelMeta RfDetrMeta(const YAML::Node& cfg) {
  Config c = ReadConfig(cfg);
  ModelMeta m;
  m.name = "rf-detr";
  m.imgsz = c.imgsz;
  m.num_classes = c.num_classes;
  m.num_queries = c.num_queries;
  m.focal = true;
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/roboflow/rf-detr";
  return m;
}

// RF-DETR-CDN: RF-DETR + contrastive denoising training (a label_enc + train-only
// ForwardDenoise via the shared deform head's CDN entry). Inference == RF-DETR.
std::shared_ptr<IModel> MakeRfDetrCdn(const YAML::Node& cfg) {
  return std::make_shared<RfDetrImpl>(ReadConfig(cfg), /*denoising=*/true);
}

ModelMeta RfDetrCdnMeta(const YAML::Node& cfg) {
  ModelMeta m = RfDetrMeta(cfg);
  m.name = "rf-detr-cdn";
  return m;
}

void RegisterRfDetr() {
  Registry::Instance().Register("rf-detr", RfDetrMeta({}), &MakeRfDetr);
  Registry::Instance().Register("rf-detr-cdn", RfDetrCdnMeta({}), &MakeRfDetrCdn);

  // The RF-DETR detection size matrix (paper Table 7, arXiv:2511.09554). n/s/m/l use
  // a DINOv2-S backbone (embed 384, depth 12, 6 heads); x uses DINOv2-B (embed 768,
  // 12 heads, patch 20). NOTE: these configs are faithful, but the ViT backbone is
  // still a structural placeholder (no DINOv2 register tokens / windowed attention),
  // so the sizes TRAIN but do not yet load official RF-DETR weights — see
  // VALIDATION.md ("registered-but-not-yet-validated").
  struct RfSize {
    const char* tag;
    int vit_embed;
    int vit_heads;
    int patch;
    int dec_layers;
    int imgsz;
  };
  constexpr RfSize kSizes[] = {
      {"n", 384, 6, 16, 2, 384}, {"s", 384, 6, 16, 3, 512}, {"m", 384, 6, 16, 4, 576},
      {"l", 384, 6, 16, 4, 704}, {"x", 768, 12, 20, 5, 700},
  };
  for (const RfSize& sz : kSizes) {
    const std::string name = std::string("rf-detr-") + sz.tag;
    auto build = [sz, name](const YAML::Node& cfg) -> std::shared_ptr<IModel> {
      Config c = ReadConfig(cfg);
      if (!(cfg && cfg["vit_embed"])) {
        c.vit_embed = sz.vit_embed;
      }
      if (!(cfg && cfg["vit_heads"])) {
        c.vit_heads = sz.vit_heads;
      }
      if (!(cfg && cfg["vit_depth"])) {
        c.vit_depth = 12;  // DINOv2-S and -B are both depth-12
      }
      if (!(cfg && cfg["patch"])) {
        c.patch = sz.patch;
      }
      if (!(cfg && cfg["dec_layers"])) {
        c.dec_layers = sz.dec_layers;
      }
      if (!(cfg && cfg["imgsz"])) {
        c.imgsz = sz.imgsz;
      }
      return std::make_shared<RfDetrImpl>(c);
    };
    ModelMeta meta;
    meta.name = name;
    meta.imgsz = sz.imgsz;
    meta.num_classes = 90;
    meta.num_queries = 300;
    meta.focal = true;
    meta.license = "Apache-2.0";
    meta.upstream = "https://github.com/roboflow/rf-detr";
    Registry::Instance().Register(name, meta, std::move(build));
  }
}

}  // namespace detr::models
