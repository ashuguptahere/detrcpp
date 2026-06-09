// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/detr.hpp"

#include <torch/torch.h>

#include <memory>
#include <string>

#include "detr/models/deformable_detr.hpp"
#include "detr/models/detr_head.hpp"
#include "detr/models/detr_r50.hpp"
#include "detr/models/model.hpp"
#include "detr/models/registry.hpp"
#include "detr/models/rt_detr.hpp"

namespace detr::models {

namespace {

namespace nn = torch::nn;

struct Config {
  int hidden_dim{256};
  int nheads{8};
  int enc_layers{6};
  int dec_layers{6};
  int dim_feedforward{2048};
  int num_queries{100};
  int num_classes{91};
  int imgsz{640};
  int backbone_width{64};
};

template <typename T>
T Get(const YAML::Node& cfg, const char* key, T fallback) {
  if (cfg && cfg[key]) {
    return cfg[key].as<T>();
  }
  return fallback;
}

Config ReadConfig(const YAML::Node& cfg) {
  Config c;
  c.hidden_dim = Get(cfg, "hidden_dim", c.hidden_dim);
  c.nheads = Get(cfg, "nheads", c.nheads);
  c.enc_layers = Get(cfg, "enc_layers", c.enc_layers);
  c.dec_layers = Get(cfg, "dec_layers", c.dec_layers);
  c.dim_feedforward = Get(cfg, "dim_feedforward", c.dim_feedforward);
  c.num_queries = Get(cfg, "num_queries", c.num_queries);
  c.num_classes = Get(cfg, "num_classes", c.num_classes);
  c.imgsz = Get(cfg, "imgsz", c.imgsz);
  c.backbone_width = Get(cfg, "backbone_width", c.backbone_width);
  return c;
}

DetrConfig ToHeadConfig(const Config& c) {
  return DetrConfig{c.hidden_dim,      c.nheads,      c.enc_layers, c.dec_layers,
                    c.dim_feedforward, c.num_queries, c.num_classes};
}

nn::Sequential MakeBackbone(int w0) {
  // A compact but real conv backbone downsampling the input by 32x. Channels
  // grow w0 -> 8*w0 across four stages after a ResNet-style stem.
  auto conv = [](int in, int out, int k, int s, int p) {
    return nn::Conv2d(nn::Conv2dOptions(in, out, k).stride(s).padding(p).bias(false));
  };
  nn::Sequential s;
  s->push_back(conv(3, w0, 7, 2, 3));
  s->push_back(nn::BatchNorm2d(w0));
  s->push_back(nn::Functional(torch::relu));
  s->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2).padding(1)));
  const int chans[4] = {w0, 2 * w0, 4 * w0, 8 * w0};
  int in = w0;
  for (int i = 0; i < 4; ++i) {
    const int out = chans[i];
    const int stride = (i == 0) ? 1 : 2;
    s->push_back(conv(in, out, 3, stride, 1));
    s->push_back(nn::BatchNorm2d(out));
    s->push_back(nn::Functional(torch::relu));
    s->push_back(conv(out, out, 3, 1, 1));
    s->push_back(nn::BatchNorm2d(out));
    s->push_back(nn::Functional(torch::relu));
    in = out;
  }
  return s;
}

class DetrImpl : public IModel {
 public:
  explicit DetrImpl(Config cfg) : cfg_(cfg) {
    backbone_ = register_module("backbone", MakeBackbone(cfg.backbone_width));
    input_proj_ = register_module(
        "input_proj", nn::Conv2d(nn::Conv2dOptions(8 * cfg.backbone_width, cfg.hidden_dim, 1)));
    head_ = BuildDetrHead(*this, ToHeadConfig(cfg));
  }

  Detections Forward(torch::Tensor images) override {
    auto feat = backbone_->forward(images);
    auto src = input_proj_->forward(feat);
    return RunDetrHead(head_, src, ToHeadConfig(cfg_));
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = "detr";
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/facebookresearch/detr";
    return m;
  }

 private:
  Config cfg_;
  nn::Sequential backbone_{nullptr};
  nn::Conv2d input_proj_{nullptr};
  DetrHead head_;
};

}  // namespace

std::shared_ptr<IModel> MakeDetr(const YAML::Node& cfg) {
  return std::make_shared<DetrImpl>(ReadConfig(cfg));
}

ModelMeta DetrMeta(const YAML::Node& cfg) {
  Config c = ReadConfig(cfg);
  ModelMeta m;
  m.name = "detr";
  m.imgsz = c.imgsz;
  m.num_classes = c.num_classes;
  m.num_queries = c.num_queries;
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/facebookresearch/detr";
  return m;
}

void RegisterBuiltins() {
  Registry::Instance().Register("detr", DetrMeta({}), &MakeDetr);
  Registry::Instance().Register("detr-r50", DetrR50Meta({}), &MakeDetrR50);
  Registry::Instance().Register("detr-r101", DetrR101Meta({}), &MakeDetrR101);
  Registry::Instance().Register("detr-r50-dc5", DetrR50Dc5Meta({}), &MakeDetrR50Dc5);
  Registry::Instance().Register("detr-r101-dc5", DetrR101Dc5Meta({}), &MakeDetrR101Dc5);
  Registry::Instance().Register("deformable-detr", DeformableDetrMeta({}), &MakeDeformableDetr);
  RegisterRtDetr();
}

}  // namespace detr::models
