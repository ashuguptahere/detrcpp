// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/rf_detr.hpp"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "detr/models/deform_attn.hpp"
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

torch::Tensor InverseSigmoid(torch::Tensor x, double eps = 1e-5) {
  x = x.clamp(0, 1);
  return torch::log(x.clamp_min(eps) / (1 - x).clamp_min(eps));
}

struct MlpImpl : nn::Module {
  nn::ModuleList layers{nullptr};
  int n_;
  MlpImpl(int in, int hidden, int out, int n) : n_(n) {
    layers = register_module("layers", nn::ModuleList());
    int prev = in;
    for (int i = 0; i < n; ++i) {
      layers->push_back(nn::Linear(prev, (i + 1 == n) ? out : hidden));
      prev = (i + 1 == n) ? out : hidden;
    }
  }
  torch::Tensor forward(torch::Tensor x) {
    for (int i = 0; i < n_; ++i) {
      x = layers[static_cast<std::size_t>(i)]->as<nn::LinearImpl>()->forward(x);
      if (i + 1 < n_) {
        x = torch::relu(x);
      }
    }
    return x;
  }
};
TORCH_MODULE(Mlp);

// Deformable decoder layer (self-attn + deformable cross-attn 4D + FFN).
struct DecoderLayerImpl : nn::Module {
  nn::MultiheadAttention self_attn{nullptr};
  MSDeformAttn cross_attn{nullptr};
  nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
  nn::Linear linear1{nullptr}, linear2{nullptr};
  DecoderLayerImpl(int d, int levels, int heads, int points, int ff) {
    self_attn = register_module(
        "self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, heads).dropout(0.0)));
    cross_attn = register_module("cross_attn", MSDeformAttn(d, levels, heads, points));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm3 = register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({d})));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
  }
  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& query_pos, const torch::Tensor& ref,
                        const torch::Tensor& memory, const SpatialShapes& shapes) {
    auto q = (tgt + query_pos).transpose(0, 1);
    auto sa = std::get<0>(self_attn->forward(q, q, tgt.transpose(0, 1))).transpose(0, 1);
    tgt = norm1->forward(tgt + sa);
    auto ca = cross_attn->forward(tgt + query_pos, ref, memory, shapes);
    tgt = norm2->forward(tgt + ca);
    auto ff = linear2->forward(torch::relu(linear1->forward(tgt)));
    return norm3->forward(tgt + ff);
  }
};
TORCH_MODULE(DecoderLayer);

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

    enc_output_ = register_module("enc_output", nn::Linear(d, d));
    enc_output_norm_ = register_module("enc_output_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
    enc_score_head_ = register_module("enc_score_head", nn::Linear(d, cfg.num_classes));
    enc_bbox_head_ = register_module("enc_bbox_head", Mlp(d, d, 4, 3));
    query_pos_head_ = register_module("query_pos_head", Mlp(4, 2 * d, d, 2));

    decoder_ = register_module("decoder", nn::ModuleList());
    dec_score_ = register_module("dec_score_head", nn::ModuleList());
    dec_bbox_ = register_module("dec_bbox_head", nn::ModuleList());
    for (int i = 0; i < cfg.dec_layers; ++i) {
      decoder_->push_back(
          DecoderLayer(d, cfg.num_levels, cfg.nheads, cfg.num_points, cfg.dim_feedforward));
      dec_score_->push_back(nn::Linear(d, cfg.num_classes));
      dec_bbox_->push_back(Mlp(d, d, 4, 3));
    }
  }

  Detections Forward(torch::Tensor images) override {
    const int d = cfg_.hidden_dim;
    const auto b = images.size(0);
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
    auto memory = torch::cat(mem, 1);  // [B, sum_hw, d]

    auto anchors = GenerateAnchors(shapes, images.options());
    auto out_mem = enc_output_norm_->forward(enc_output_->forward(memory));
    auto enc_class = enc_score_head_->forward(out_mem);
    auto enc_coord = enc_bbox_head_->forward(out_mem) + anchors;

    auto topk = std::get<1>(std::get<0>(enc_class.max(-1)).topk(cfg_.num_queries, 1));
    auto gi = topk.unsqueeze(-1);
    auto ref_unact = enc_coord.gather(1, gi.expand({b, cfg_.num_queries, 4})).detach();
    auto tgt = out_mem.gather(1, gi.expand({b, cfg_.num_queries, d})).detach();

    auto ref = ref_unact.sigmoid();
    torch::Tensor logits;
    torch::Tensor boxes;
    for (int i = 0; i < cfg_.dec_layers; ++i) {
      auto ref_input = ref.unsqueeze(2).expand({b, cfg_.num_queries, cfg_.num_levels, 4});
      auto query_pos = query_pos_head_->forward(ref);
      tgt = decoder_[static_cast<std::size_t>(i)]->as<DecoderLayerImpl>()->forward(
          tgt, query_pos, ref_input, memory, shapes);
      auto bbox = (dec_bbox_[static_cast<std::size_t>(i)]->as<MlpImpl>()->forward(tgt) +
                   InverseSigmoid(ref))
                      .sigmoid();
      logits = dec_score_[static_cast<std::size_t>(i)]->as<nn::LinearImpl>()->forward(tgt);
      boxes = bbox;
      ref = bbox.detach();
    }

    Detections det;
    det.logits = logits;
    det.boxes = boxes;
    return det;
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
  torch::Tensor GenerateAnchors(const SpatialShapes& shapes, const torch::TensorOptions& opts) {
    std::vector<torch::Tensor> anchors;
    int lvl = 0;
    for (const auto& [h, w] : shapes) {
      auto grid = torch::meshgrid({torch::arange(h, opts), torch::arange(w, opts)}, "ij");
      auto xy = torch::stack({grid[1], grid[0]}, -1);
      auto wht = torch::tensor({static_cast<double>(w), static_cast<double>(h)}, opts);
      auto xyn = (xy.unsqueeze(0) + 0.5) / wht;
      auto wh = torch::ones_like(xyn) * (0.05 * std::pow(2.0, lvl));
      anchors.push_back(torch::cat({xyn, wh}, -1).reshape({1, h * w, 4}));
      ++lvl;
    }
    auto a = torch::cat(anchors, 1);
    auto valid = ((a > 1e-2) * (a < 1 - 1e-2)).all(-1, true);
    a = torch::log(a / (1 - a));
    return torch::where(valid.expand_as(a), a, torch::full_like(a, 1e9));
  }

  Config cfg_;
  ViT backbone_{nullptr};
  nn::ModuleList input_proj_{nullptr};
  nn::Linear enc_output_{nullptr};
  nn::LayerNorm enc_output_norm_{nullptr};
  nn::Linear enc_score_head_{nullptr};
  Mlp enc_bbox_head_{nullptr};
  Mlp query_pos_head_{nullptr};
  nn::ModuleList decoder_{nullptr};
  nn::ModuleList dec_score_{nullptr};
  nn::ModuleList dec_bbox_{nullptr};
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
