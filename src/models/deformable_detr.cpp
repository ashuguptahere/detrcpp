// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/deformable_detr.hpp"

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "detr/models/deform_attn.hpp"
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
  int dim_feedforward{1024};
  int num_queries{300};
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

// PositionEmbeddingSine (Deformable-DETR), no mask: returns [B, d, H, W].
torch::Tensor SinePos(std::int64_t b, std::int64_t d, std::int64_t h, std::int64_t w,
                      const torch::TensorOptions& opts) {
  constexpr double kPi = 3.14159265358979323846;
  const std::int64_t half = d / 2;
  const double scale = 2.0 * kPi;
  auto y_embed = torch::arange(1, h + 1, opts).view({h, 1});
  auto x_embed = torch::arange(1, w + 1, opts).view({1, w});
  y_embed = y_embed / (y_embed[h - 1] + 1e-6) * scale;
  x_embed = x_embed / (x_embed.index({0, w - 1}) + 1e-6) * scale;
  auto dim_t = torch::arange(0, half, opts);
  dim_t = torch::pow(10000.0, 2 * torch::floor(dim_t / 2) / static_cast<double>(half));
  auto interleave = [half](torch::Tensor p) {  // p: [..., half]
    return torch::stack({p.slice(-1, 0, half, 2).sin(), p.slice(-1, 1, half, 2).cos()}, -1)
        .flatten(-2);
  };
  auto pos_x = interleave(x_embed.unsqueeze(-1) / dim_t);  // [1, w, half]
  auto pos_y = interleave(y_embed.unsqueeze(-1) / dim_t);  // [h, 1, half]
  auto pyf = pos_y.expand({h, w, half});
  auto pxf = pos_x.expand({h, w, half});
  auto pos = torch::cat({pyf, pxf}, 2);  // [h, w, d]
  return pos.permute({2, 0, 1}).unsqueeze(0).expand({b, d, h, w}).contiguous();
}

torch::Tensor InverseSigmoid(torch::Tensor x, double eps = 1e-5) {
  x = x.clamp(0, 1);
  return torch::log(x.clamp_min(eps) / (1 - x).clamp_min(eps));
}

// Normalized-center reference grid for the encoder: [1, Sum(H*W), n_levels, 2].
torch::Tensor EncoderReferencePoints(const SpatialShapes& shapes,
                                     const torch::TensorOptions& opts) {
  std::vector<torch::Tensor> refs;
  for (const auto& [h, w] : shapes) {
    auto ry = (torch::arange(h, opts) + 0.5) / static_cast<double>(h);
    auto rx = (torch::arange(w, opts) + 0.5) / static_cast<double>(w);
    auto grid = torch::meshgrid({ry, rx}, "ij");
    auto ref = torch::stack({grid[1].reshape(-1), grid[0].reshape(-1)}, -1);  // (x, y)
    refs.push_back(ref);
  }
  auto reference = torch::cat(refs, 0);  // [Sum(H*W), 2]
  const auto levels = static_cast<std::int64_t>(shapes.size());
  return reference.view({1, -1, 1, 2}).expand({1, reference.size(0), levels, 2}).contiguous();
}

struct EncoderLayerImpl : nn::Module {
  MSDeformAttn self_attn{nullptr};
  nn::LayerNorm norm1{nullptr};
  nn::Linear linear1{nullptr};
  nn::Linear linear2{nullptr};
  nn::LayerNorm norm2{nullptr};

  EncoderLayerImpl(int d, int levels, int heads, int points, int ff) {
    self_attn = register_module("self_attn", MSDeformAttn(d, levels, heads, points));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
  }

  torch::Tensor forward(torch::Tensor src, const torch::Tensor& pos, const torch::Tensor& ref,
                        const SpatialShapes& shapes) {
    auto src2 = self_attn->forward(src + pos, ref, src, shapes);
    src = norm1->forward(src + src2);
    auto ffn = linear2->forward(torch::relu(linear1->forward(src)));
    return norm2->forward(src + ffn);
  }
};
TORCH_MODULE(EncoderLayer);

struct DecoderLayerImpl : nn::Module {
  nn::MultiheadAttention self_attn{nullptr};
  MSDeformAttn cross_attn{nullptr};
  nn::LayerNorm norm1{nullptr};
  nn::LayerNorm norm2{nullptr};
  nn::LayerNorm norm3{nullptr};
  nn::Linear linear1{nullptr};
  nn::Linear linear2{nullptr};

  DecoderLayerImpl(int d, int levels, int heads, int points, int ff) {
    self_attn = register_module(
        "self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, heads).dropout(0.1)));
    cross_attn = register_module("cross_attn", MSDeformAttn(d, levels, heads, points));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm3 = register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({d})));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
  }

  torch::Tensor forward(torch::Tensor tgt, const torch::Tensor& query_pos, const torch::Tensor& ref,
                        const torch::Tensor& memory, const SpatialShapes& shapes) {
    auto q = (tgt + query_pos).transpose(0, 1);  // [Lq, N, d] for nn::MHA
    auto k = q;
    auto v = tgt.transpose(0, 1);
    auto sa = std::get<0>(self_attn->forward(q, k, v)).transpose(0, 1);
    tgt = norm1->forward(tgt + sa);
    auto ca = cross_attn->forward(tgt + query_pos, ref, memory, shapes);
    tgt = norm2->forward(tgt + ca);
    auto ffn = linear2->forward(torch::relu(linear1->forward(tgt)));
    return norm3->forward(tgt + ffn);
  }
};
TORCH_MODULE(DecoderLayer);

class DeformableDetrImpl : public IModel {
 public:
  explicit DeformableDetrImpl(Config cfg) : cfg_(cfg) {
    const int d = cfg.hidden_dim;
    backbone_ = register_module(
        "backbone", ResNet(std::vector<int>{3, 4, 6, 3}, /*bottleneck=*/true, /*dc5=*/false));

    // input_proj: 1x1 on C3/C4/C5 (+ GroupNorm), then 3x3 stride-2 extra levels.
    input_proj_ = register_module("input_proj", nn::ModuleList());
    const int backbone_ch[3] = {512, 1024, 2048};
    const int backbone_outs = 3;
    for (int i = 0; i < backbone_outs; ++i) {
      input_proj_->push_back(nn::Sequential(nn::Conv2d(nn::Conv2dOptions(backbone_ch[i], d, 1)),
                                            nn::GroupNorm(nn::GroupNormOptions(32, d))));
    }
    int in_ch = 2048;
    for (int i = backbone_outs; i < cfg.num_levels; ++i) {
      input_proj_->push_back(
          nn::Sequential(nn::Conv2d(nn::Conv2dOptions(in_ch, d, 3).stride(2).padding(1)),
                         nn::GroupNorm(nn::GroupNormOptions(32, d))));
      in_ch = d;
    }

    level_embed_ = register_parameter("level_embed", torch::randn({cfg.num_levels, d}));
    query_embed_ = register_module("query_embed", nn::Embedding(cfg.num_queries, d * 2));
    reference_points_ = register_module("reference_points", nn::Linear(d, 2));

    encoder_ = register_module("encoder", nn::ModuleList());
    for (int i = 0; i < cfg.enc_layers; ++i) {
      encoder_->push_back(
          EncoderLayer(d, cfg.num_levels, cfg.nheads, cfg.num_points, cfg.dim_feedforward));
    }
    decoder_ = register_module("decoder", nn::ModuleList());
    for (int i = 0; i < cfg.dec_layers; ++i) {
      decoder_->push_back(
          DecoderLayer(d, cfg.num_levels, cfg.nheads, cfg.num_points, cfg.dim_feedforward));
    }

    // Deformable-DETR uses a sigmoid/focal classifier: num_classes logits (no
    // explicit no-object slot), and a 3-layer MLP box head over reference points.
    class_embed_ = register_module("class_embed", nn::Linear(d, cfg.num_classes));
    bbox_embed_ =
        register_module("bbox_embed", nn::Sequential(nn::Linear(d, d), nn::Functional(torch::relu),
                                                     nn::Linear(d, d), nn::Functional(torch::relu),
                                                     nn::Linear(d, 4)));
  }

  Detections Forward(torch::Tensor images) override {
    const int d = cfg_.hidden_dim;
    const auto b = images.size(0);
    auto feats = backbone_->forward_features(images);  // {C3, C4, C5}

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
    std::vector<torch::Tensor> src_flat;
    std::vector<torch::Tensor> pos_flat;
    for (int l = 0; l < cfg_.num_levels; ++l) {
      auto src = srcs[static_cast<std::size_t>(l)];
      const auto h = src.size(2);
      const auto w = src.size(3);
      shapes.emplace_back(h, w);
      auto pos = SinePos(b, d, h, w, src.options()).flatten(2).transpose(1, 2);  // [B, hw, d]
      auto lvl = pos + level_embed_[l].view({1, 1, -1});
      src_flat.push_back(src.flatten(2).transpose(1, 2));
      pos_flat.push_back(lvl);
    }
    auto src_cat = torch::cat(src_flat, 1);  // [B, Sum(hw), d]
    auto pos_cat = torch::cat(pos_flat, 1);
    auto enc_ref = EncoderReferencePoints(shapes, images.options()).expand({b, -1, -1, -1});

    auto memory = src_cat;
    for (const auto& m : *encoder_) {
      memory = m->as<EncoderLayerImpl>()->forward(memory, pos_cat, enc_ref, shapes);
    }

    auto qe = query_embed_->weight;                                       // [num_queries, 2d]
    auto query_pos = qe.slice(1, 0, d).unsqueeze(0).expand({b, -1, -1});  // [B, nq, d]
    auto tgt = qe.slice(1, d, 2 * d).unsqueeze(0).expand({b, -1, -1});
    auto reference_points = reference_points_->forward(query_pos).sigmoid();  // [B, nq, 2]
    auto dec_ref = reference_points.unsqueeze(2).expand({b, -1, cfg_.num_levels, 2});

    auto out = tgt;
    for (const auto& m : *decoder_) {
      out = m->as<DecoderLayerImpl>()->forward(out, query_pos, dec_ref, memory, shapes);
    }

    Detections det;
    det.logits = class_embed_->forward(out);  // [B, nq, num_classes] (sigmoid/focal)
    auto box = bbox_embed_->forward(out);     // [B, nq, 4]
    box = box + torch::cat({InverseSigmoid(reference_points),
                            torch::zeros({b, cfg_.num_queries, 2}, box.options())},
                           -1);
    det.boxes = box.sigmoid();
    return det;
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = "deformable-detr";
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.focal = true;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/fundamentalvision/Deformable-DETR";
    return m;
  }

 private:
  Config cfg_;
  ResNet backbone_{nullptr};
  nn::ModuleList input_proj_{nullptr};
  torch::Tensor level_embed_;
  nn::Embedding query_embed_{nullptr};
  nn::Linear reference_points_{nullptr};
  nn::ModuleList encoder_{nullptr};
  nn::ModuleList decoder_{nullptr};
  nn::Linear class_embed_{nullptr};
  nn::Sequential bbox_embed_{nullptr};
};

}  // namespace

std::shared_ptr<IModel> MakeDeformableDetr(const YAML::Node& cfg) {
  return std::make_shared<DeformableDetrImpl>(ReadConfig(cfg));
}

ModelMeta DeformableDetrMeta(const YAML::Node& cfg) {
  Config c = ReadConfig(cfg);
  ModelMeta m;
  m.name = "deformable-detr";
  m.imgsz = c.imgsz;
  m.num_classes = c.num_classes;
  m.num_queries = c.num_queries;
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/fundamentalvision/Deformable-DETR";
  return m;
}

}  // namespace detr::models
