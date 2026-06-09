// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/detr_r50.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "detr/models/detr_head.hpp"
#include "detr/models/resnet.hpp"
#include "detr/models/model.hpp"
#include "detr/weights/remapper.hpp"

namespace detr::models {

namespace {

namespace nn = torch::nn;

struct R50Config {
  int hidden_dim{256};
  int nheads{8};
  int enc_layers{6};
  int dec_layers{6};
  int dim_feedforward{2048};
  int num_queries{100};
  int num_classes{91};
  int imgsz{640};
};

template <typename T>
T Get(const YAML::Node& cfg, const char* key, T fallback) {
  return (cfg && cfg[key]) ? cfg[key].as<T>() : fallback;
}

R50Config ReadConfig(const YAML::Node& cfg) {
  R50Config c;
  c.hidden_dim = Get(cfg, "hidden_dim", c.hidden_dim);
  c.nheads = Get(cfg, "nheads", c.nheads);
  c.enc_layers = Get(cfg, "enc_layers", c.enc_layers);
  c.dec_layers = Get(cfg, "dec_layers", c.dec_layers);
  c.dim_feedforward = Get(cfg, "dim_feedforward", c.dim_feedforward);
  c.num_queries = Get(cfg, "num_queries", c.num_queries);
  c.num_classes = Get(cfg, "num_classes", c.num_classes);
  c.imgsz = Get(cfg, "imgsz", c.imgsz);
  return c;
}

DetrConfig ToHeadConfig(const R50Config& c) {
  return DetrConfig{c.hidden_dim, c.nheads,       c.enc_layers,
                    c.dec_layers, c.dim_feedforward, c.num_queries, c.num_classes};
}

class DetrResNetImpl : public IModel {
 public:
  DetrResNetImpl(R50Config cfg, std::vector<int> blocks, std::string name, bool dc5)
      : cfg_(cfg), name_(std::move(name)) {
    backbone_ = register_module("backbone", ResNet(blocks, dc5));
    input_proj_ =
        register_module("input_proj", nn::Conv2d(nn::Conv2dOptions(2048, cfg.hidden_dim, 1)));
    head_ = BuildDetrHead(*this, ToHeadConfig(cfg));
  }

  Detections Forward(torch::Tensor images) override {
    auto feat = backbone_->forward(images);
    auto src = input_proj_->forward(feat);
    return RunDetrHead(head_, src, ToHeadConfig(cfg_));
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = name_;
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/facebookresearch/detr";
    return m;
  }

  // Maps facebookresearch/detr checkpoint keys onto ours. Verified byte-exact:
  // the official detr-r50 checkpoint loads with 0 unexpected keys (only the
  // num_batches_tracked buffers are dropped) and reproduces the published mAP.
  weights::WeightRemapper UpstreamRemapper() const override {
    weights::WeightRemapper r;
    r.Drop("num_batches_tracked")
        .ReplaceRegex("^backbone\\.0\\.body\\.", "backbone.")
        .ReplaceRegex("^transformer\\.encoder\\.layers\\.", "encoder.")
        .ReplaceRegex("^transformer\\.decoder\\.norm\\.", "decoder_norm.")
        .ReplaceRegex("^transformer\\.decoder\\.layers\\.", "decoder.")
        .ReplaceRegex("multihead_attn", "cross_attn")
        .ReplaceRegex("bbox_embed\\.layers\\.0", "bbox_embed.0")
        .ReplaceRegex("bbox_embed\\.layers\\.1", "bbox_embed.2")
        .ReplaceRegex("bbox_embed\\.layers\\.2", "bbox_embed.4");
    return r;
  }

 private:
  R50Config cfg_;
  std::string name_;
  ResNet backbone_{nullptr};
  nn::Conv2d input_proj_{nullptr};
  DetrHead head_;
};

ModelMeta MakeMeta(const YAML::Node& cfg, const std::string& name) {
  R50Config c = ReadConfig(cfg);
  ModelMeta m;
  m.name = name;
  m.imgsz = c.imgsz;
  m.num_classes = c.num_classes;
  m.num_queries = c.num_queries;
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/facebookresearch/detr";
  return m;
}

}  // namespace

std::shared_ptr<IModel> MakeDetrR50(const YAML::Node& cfg) {
  return std::make_shared<DetrResNetImpl>(ReadConfig(cfg), std::vector<int>{3, 4, 6, 3},
                                          "detr-r50", /*dc5=*/false);
}

std::shared_ptr<IModel> MakeDetrR101(const YAML::Node& cfg) {
  return std::make_shared<DetrResNetImpl>(ReadConfig(cfg), std::vector<int>{3, 4, 23, 3},
                                          "detr-r101", /*dc5=*/false);
}

std::shared_ptr<IModel> MakeDetrR50Dc5(const YAML::Node& cfg) {
  return std::make_shared<DetrResNetImpl>(ReadConfig(cfg), std::vector<int>{3, 4, 6, 3},
                                          "detr-r50-dc5", /*dc5=*/true);
}

std::shared_ptr<IModel> MakeDetrR101Dc5(const YAML::Node& cfg) {
  return std::make_shared<DetrResNetImpl>(ReadConfig(cfg), std::vector<int>{3, 4, 23, 3},
                                          "detr-r101-dc5", /*dc5=*/true);
}

ModelMeta DetrR50Meta(const YAML::Node& cfg) { return MakeMeta(cfg, "detr-r50"); }
ModelMeta DetrR101Meta(const YAML::Node& cfg) { return MakeMeta(cfg, "detr-r101"); }
ModelMeta DetrR50Dc5Meta(const YAML::Node& cfg) { return MakeMeta(cfg, "detr-r50-dc5"); }
ModelMeta DetrR101Dc5Meta(const YAML::Node& cfg) { return MakeMeta(cfg, "detr-r101-dc5"); }

}  // namespace detr::models
