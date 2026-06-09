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
  explicit RfDetrImpl(Config cfg) : cfg_(cfg) {
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
  }

  Detections Forward(torch::Tensor images) override {
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
    return RunDeformDetectHead(head_, torch::cat(mem, 1), shapes);
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = "rf-detr";
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.focal = true;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/roboflow/rf-detr";
    return m;
  }

 private:
  Config cfg_;
  ViT backbone_{nullptr};
  nn::ModuleList input_proj_{nullptr};
  DeformDetectHead head_;
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

}  // namespace detr::models
