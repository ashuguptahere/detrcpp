// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/dino.hpp"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "detr/models/deform_attn.hpp"
#include "detr/models/deform_head.hpp"
#include "detr/models/model.hpp"
#include "detr/models/registry.hpp"
#include "detr/models/resnet.hpp"

namespace detr::models {

namespace {

namespace nn = torch::nn;

struct Config {
  int hidden_dim{256};
  int nheads{8};
  int enc_layers{6};
  int dec_layers{6};
  int dim_feedforward{2048};
  int num_queries{900};
  int num_classes{91};
  int num_levels{4};
  int num_points{4};
  int imgsz{640};
};

template <typename T>
T Get(const YAML::Node& c, const char* k, T fb) {
  return (c && c[k]) ? c[k].as<T>() : fb;
}

Config ReadConfig(const YAML::Node& c) {
  Config x;
  x.hidden_dim = Get(c, "hidden_dim", x.hidden_dim);
  x.nheads = Get(c, "nheads", x.nheads);
  x.enc_layers = Get(c, "enc_layers", x.enc_layers);
  x.dec_layers = Get(c, "dec_layers", x.dec_layers);
  x.dim_feedforward = Get(c, "dim_feedforward", x.dim_feedforward);
  x.num_queries = Get(c, "num_queries", x.num_queries);
  x.num_classes = Get(c, "num_classes", x.num_classes);
  x.num_levels = Get(c, "num_levels", x.num_levels);
  x.num_points = Get(c, "num_points", x.num_points);
  x.imgsz = Get(c, "imgsz", x.imgsz);
  return x;
}

torch::Tensor SinePos(std::int64_t b, std::int64_t d, std::int64_t h, std::int64_t w,
                      const torch::TensorOptions& opts) {
  constexpr double kPi = 3.14159265358979323846;
  const std::int64_t half = d / 2;
  const double scale = 2.0 * kPi;
  auto ys = torch::arange(1, h + 1, opts) / (static_cast<double>(h) + 1e-6) * scale;
  auto xs = torch::arange(1, w + 1, opts) / (static_cast<double>(w) + 1e-6) * scale;
  auto dim_t = torch::arange(0, half, opts);
  dim_t = torch::pow(10000.0, 2 * torch::floor(dim_t / 2) / static_cast<double>(half));
  auto px = xs.unsqueeze(1) / dim_t.unsqueeze(0);
  auto py = ys.unsqueeze(1) / dim_t.unsqueeze(0);
  auto interleave = [half](torch::Tensor p) {
    return torch::stack({p.slice(1, 0, half, 2).sin(), p.slice(1, 1, half, 2).cos()}, 2).flatten(1);
  };
  px = interleave(px);
  py = interleave(py);
  auto pyf = py.unsqueeze(1).expand({h, w, half});
  auto pxf = px.unsqueeze(0).expand({h, w, half});
  auto pos = torch::cat({pyf, pxf}, 2);
  return pos.permute({2, 0, 1}).unsqueeze(0).expand({b, d, h, w}).contiguous();
}

torch::Tensor EncoderRef(const SpatialShapes& shapes, std::int64_t levels,
                         const torch::TensorOptions& opts) {
  std::vector<torch::Tensor> refs;
  for (const auto& [h, w] : shapes) {
    auto ry = (torch::arange(h, opts) + 0.5) / static_cast<double>(h);
    auto rx = (torch::arange(w, opts) + 0.5) / static_cast<double>(w);
    auto grid = torch::meshgrid({ry, rx}, "ij");
    refs.push_back(torch::stack({grid[1].reshape(-1), grid[0].reshape(-1)}, -1));
  }
  auto r = torch::cat(refs, 0);
  return r.view({1, -1, 1, 2}).expand({1, r.size(0), levels, 2}).contiguous();
}

struct EncoderLayerImpl : nn::Module {
  MSDeformAttn self_attn{nullptr};
  nn::LayerNorm norm1{nullptr}, norm2{nullptr};
  nn::Linear linear1{nullptr}, linear2{nullptr};
  EncoderLayerImpl(int d, int levels, int heads, int points, int ff) {
    self_attn = register_module("self_attn", MSDeformAttn(d, levels, heads, points));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
  }
  torch::Tensor forward(torch::Tensor src, const torch::Tensor& pos, const torch::Tensor& ref,
                        const SpatialShapes& shapes) {
    auto s = self_attn->forward(src + pos, ref, src, shapes);
    src = norm1->forward(src + s);
    auto ff = linear2->forward(torch::relu(linear1->forward(src)));
    return norm2->forward(src + ff);
  }
};
TORCH_MODULE(EncoderLayer);

class DinoImpl : public IModel {
 public:
  explicit DinoImpl(Config cfg) : cfg_(cfg) {
    const int d = cfg.hidden_dim;
    backbone_ = register_module(
        "backbone", ResNet(std::vector<int>{3, 4, 6, 3}, /*bottleneck=*/true, /*dc5=*/false));
    const int ch[3] = {512, 1024, 2048};
    input_proj_ = register_module("input_proj", nn::ModuleList());
    for (int i = 0; i < 3; ++i) {
      input_proj_->push_back(nn::Sequential(nn::Conv2d(nn::Conv2dOptions(ch[i], d, 1)),
                                            nn::GroupNorm(nn::GroupNormOptions(32, d))));
    }
    int in_ch = 2048;
    for (int i = 3; i < cfg.num_levels; ++i) {
      input_proj_->push_back(
          nn::Sequential(nn::Conv2d(nn::Conv2dOptions(in_ch, d, 3).stride(2).padding(1)),
                         nn::GroupNorm(nn::GroupNormOptions(32, d))));
      in_ch = d;
    }
    level_embed_ = register_parameter("level_embed", torch::randn({cfg.num_levels, d}));

    encoder_ = register_module("encoder", nn::ModuleList());
    for (int i = 0; i < cfg.enc_layers; ++i) {
      encoder_->push_back(
          EncoderLayer(d, cfg.num_levels, cfg.nheads, cfg.num_points, cfg.dim_feedforward));
    }
    head_ = BuildDeformDetectHead(*this, d, cfg.num_levels, cfg.nheads, cfg.num_points,
                                  cfg.dim_feedforward, cfg.dec_layers, cfg.num_classes,
                                  cfg.num_queries);
  }

  Detections Forward(torch::Tensor images) override {
    const int d = cfg_.hidden_dim;
    const auto b = images.size(0);
    auto feats = backbone_->forward_features(images);

    std::vector<torch::Tensor> srcs;
    for (int i = 0; i < 3; ++i) {
      srcs.push_back(input_proj_[static_cast<std::size_t>(i)]->as<nn::SequentialImpl>()->forward(
          feats[static_cast<std::size_t>(i)]));
    }
    for (int i = 3; i < cfg_.num_levels; ++i) {
      auto in = (i == 3) ? feats[2] : srcs.back();
      srcs.push_back(
          input_proj_[static_cast<std::size_t>(i)]->as<nn::SequentialImpl>()->forward(in));
    }

    SpatialShapes shapes;
    std::vector<torch::Tensor> src_flat, pos_flat;
    for (int l = 0; l < cfg_.num_levels; ++l) {
      auto s = srcs[static_cast<std::size_t>(l)];
      shapes.emplace_back(s.size(2), s.size(3));
      auto pos = SinePos(b, d, s.size(2), s.size(3), s.options()).flatten(2).transpose(1, 2);
      src_flat.push_back(s.flatten(2).transpose(1, 2));
      pos_flat.push_back(pos + level_embed_[l].view({1, 1, -1}));
    }
    auto src = torch::cat(src_flat, 1);
    auto pos = torch::cat(pos_flat, 1);
    auto enc_ref = EncoderRef(shapes, cfg_.num_levels, images.options()).expand({b, -1, -1, -1});

    auto memory = src;
    for (const auto& m : *encoder_) {
      memory = m->as<EncoderLayerImpl>()->forward(memory, pos, enc_ref, shapes);
    }
    return RunDeformDetectHead(head_, memory, shapes);
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = "dino";
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.focal = true;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/IDEA-Research/DINO";
    return m;
  }

 private:
  Config cfg_;
  ResNet backbone_{nullptr};
  nn::ModuleList input_proj_{nullptr};
  torch::Tensor level_embed_;
  nn::ModuleList encoder_{nullptr};
  DeformDetectHead head_;
};

}  // namespace

std::shared_ptr<IModel> MakeDino(const YAML::Node& cfg) {
  return std::make_shared<DinoImpl>(ReadConfig(cfg));
}

ModelMeta DinoMeta(const YAML::Node& cfg) {
  Config c = ReadConfig(cfg);
  ModelMeta m;
  m.name = "dino";
  m.imgsz = c.imgsz;
  m.num_classes = c.num_classes;
  m.num_queries = c.num_queries;
  m.focal = true;
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/IDEA-Research/DINO";
  return m;
}

}  // namespace detr::models
