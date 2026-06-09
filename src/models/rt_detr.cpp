// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/rt_detr.hpp"

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
namespace F = torch::nn::functional;

struct Config {
  std::string backbone{"r50"};  // r18 / r34 / r50 / r101
  std::string name{"rt-detr"};  // registry name (rt-detr[vN]-{size})
  int hidden_dim{256};
  int nheads{8};
  int enc_layers{1};  // AIFI layers (on the top level only)
  int dec_layers{6};
  int dim_feedforward{1024};
  int num_queries{300};
  int num_classes{80};
  int num_levels{3};
  int num_points{4};
  int imgsz{640};
};

struct BackboneSpec {
  std::vector<int> blocks;
  bool bottleneck;
};

BackboneSpec BackboneFor(const std::string& name) {
  if (name == "r18") {
    return {{2, 2, 2, 2}, false};
  }
  if (name == "r34") {
    return {{3, 4, 6, 3}, false};
  }
  if (name == "r101") {
    return {{3, 4, 23, 3}, true};
  }
  return {{3, 4, 6, 3}, true};  // r50
}

template <typename T>
T Get(const YAML::Node& c, const char* k, T fb) {
  return (c && c[k]) ? c[k].as<T>() : fb;
}

Config ReadConfig(const YAML::Node& c) {
  Config x;
  x.backbone = Get<std::string>(c, "backbone", x.backbone);
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

torch::Tensor InverseSigmoid(torch::Tensor x, double eps = 1e-5) {
  x = x.clamp(0, 1);
  return torch::log(x.clamp_min(eps) / (1 - x).clamp_min(eps));
}

// 2D sin-cos positional embedding for AIFI: [1, H*W, dim].
torch::Tensor SinCos2d(std::int64_t h, std::int64_t w, int dim, const torch::TensorOptions& opts,
                       double temp = 10000.0) {
  auto gw = torch::arange(w, opts);
  auto gh = torch::arange(h, opts);
  auto grid = torch::meshgrid({gw, gh}, "ij");  // grid[0]=w-idx, grid[1]=h-idx
  const int pos_dim = dim / 4;
  auto omega = torch::arange(pos_dim, opts) / static_cast<double>(pos_dim);
  omega = 1.0 / torch::pow(temp, omega);
  auto out_w = grid[0].flatten().unsqueeze(1) * omega.unsqueeze(0);  // [w*h, pos_dim]
  auto out_h = grid[1].flatten().unsqueeze(1) * omega.unsqueeze(0);
  auto pos = torch::cat({out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()}, 1);
  return pos.unsqueeze(0);  // [1, w*h, dim]
}

// Conv2d(no bias) + BatchNorm + optional SiLU.
struct ConvNormImpl : nn::Module {
  nn::Conv2d conv{nullptr};
  nn::BatchNorm2d norm{nullptr};
  bool act_;
  ConvNormImpl(int in, int out, int k, int s, bool act) : act_(act) {
    conv = register_module(
        "conv",
        nn::Conv2d(nn::Conv2dOptions(in, out, k).stride(s).padding((k - 1) / 2).bias(false)));
    norm = register_module("norm", nn::BatchNorm2d(out));
  }
  torch::Tensor forward(torch::Tensor x) {
    x = norm->forward(conv->forward(x));
    return act_ ? torch::silu(x) : x;
  }
};
TORCH_MODULE(ConvNorm);

// RepVGG block (training-time two-branch form): 3x3 + 1x1, SiLU.
struct RepVggImpl : nn::Module {
  ConvNorm conv1{nullptr};
  ConvNorm conv2{nullptr};
  RepVggImpl(int c) {
    conv1 = register_module("conv1", ConvNorm(c, c, 3, 1, /*act=*/false));
    conv2 = register_module("conv2", ConvNorm(c, c, 1, 1, /*act=*/false));
  }
  torch::Tensor forward(torch::Tensor x) {
    return torch::silu(conv1->forward(x) + conv2->forward(x));
  }
};
TORCH_MODULE(RepVgg);

// CSPRepLayer: two 1x1 branches, RepVGG bottlenecks on one, fused by a 1x1.
struct CSPRepImpl : nn::Module {
  ConvNorm conv1{nullptr};
  ConvNorm conv2{nullptr};
  nn::Sequential bottlenecks{nullptr};
  ConvNorm conv3{nullptr};
  bool identity_;
  CSPRepImpl(int in, int out, int num_blocks) : identity_(in == out) {
    conv1 = register_module("conv1", ConvNorm(in, out, 1, 1, true));
    conv2 = register_module("conv2", ConvNorm(in, out, 1, 1, true));
    bottlenecks = nn::Sequential();
    for (int i = 0; i < num_blocks; ++i) {
      bottlenecks->push_back(RepVgg(out));
    }
    register_module("bottlenecks", bottlenecks);
    if (!identity_) {
      conv3 = register_module("conv3", ConvNorm(out, out, 1, 1, true));
    }
  }
  torch::Tensor forward(torch::Tensor x) {
    auto x1 = bottlenecks->forward(conv1->forward(x));
    auto x2 = conv2->forward(x);
    auto out = x1 + x2;
    return identity_ ? out : conv3->forward(out);
  }
};
TORCH_MODULE(CSPRep);

// AIFI: a standard post-norm transformer encoder layer (used on the top level).
struct AIFILayerImpl : nn::Module {
  nn::MultiheadAttention self_attn{nullptr};
  nn::Linear linear1{nullptr};
  nn::Linear linear2{nullptr};
  nn::LayerNorm norm1{nullptr};
  nn::LayerNorm norm2{nullptr};
  AIFILayerImpl(int d, int heads, int ff) {
    self_attn = register_module(
        "self_attn", nn::MultiheadAttention(nn::MultiheadAttentionOptions(d, heads).dropout(0.0)));
    linear1 = register_module("linear1", nn::Linear(d, ff));
    linear2 = register_module("linear2", nn::Linear(ff, d));
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({d})));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({d})));
  }
  torch::Tensor forward(torch::Tensor src, const torch::Tensor& pos) {
    auto q = (src + pos).transpose(0, 1);  // [L, N, d]
    auto k = q;
    auto v = src.transpose(0, 1);
    auto attn = std::get<0>(self_attn->forward(q, k, v)).transpose(0, 1);
    src = norm1->forward(src + attn);
    auto ff = linear2->forward(torch::relu(linear1->forward(src)));
    return norm2->forward(src + ff);
  }
};
TORCH_MODULE(AIFILayer);

// Decoder layer: self-attn + deformable cross-attn (4D refs) + FFN.
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

// 3-layer MLP (ReLU between).
struct MlpImpl : nn::Module {
  nn::ModuleList layers{nullptr};
  int n_;
  MlpImpl(int in, int hidden, int out, int n) : n_(n) {
    layers = register_module("layers", nn::ModuleList());
    int prev = in;
    for (int i = 0; i < n; ++i) {
      int o = (i + 1 == n) ? out : hidden;
      layers->push_back(nn::Linear(prev, o));
      prev = o;
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

class RtDetrImpl : public IModel {
 public:
  explicit RtDetrImpl(Config cfg) : cfg_(cfg) {
    const int d = cfg.hidden_dim;
    auto spec = BackboneFor(cfg.backbone);
    backbone_ = register_module("backbone", ResNet(spec.blocks, spec.bottleneck, /*dc5=*/false));
    const auto backbone_ch = backbone_->feature_channels();  // {C3, C4, C5}

    // input_proj: 1x1 conv + BN to hidden, per backbone level (C3,C4,C5).
    input_proj_ = register_module("input_proj", nn::ModuleList());
    for (int i = 0; i < cfg.num_levels; ++i) {
      input_proj_->push_back(nn::Sequential(
          nn::Conv2d(nn::Conv2dOptions(backbone_ch[static_cast<std::size_t>(i)], d, 1).bias(false)),
          nn::BatchNorm2d(d)));
    }

    // AIFI on the top level.
    aifi_ = register_module("aifi", nn::ModuleList());
    for (int i = 0; i < cfg.enc_layers; ++i) {
      aifi_->push_back(AIFILayer(d, cfg.nheads, cfg.dim_feedforward));
    }

    // CCFM top-down (FPN) + bottom-up (PAN).
    lateral_convs_ = register_module("lateral_convs", nn::ModuleList());
    fpn_blocks_ = register_module("fpn_blocks", nn::ModuleList());
    for (int i = 0; i < cfg.num_levels - 1; ++i) {
      lateral_convs_->push_back(ConvNorm(d, d, 1, 1, true));
      fpn_blocks_->push_back(CSPRep(d * 2, d, 3));
    }
    downsample_convs_ = register_module("downsample_convs", nn::ModuleList());
    pan_blocks_ = register_module("pan_blocks", nn::ModuleList());
    for (int i = 0; i < cfg.num_levels - 1; ++i) {
      downsample_convs_->push_back(ConvNorm(d, d, 3, 2, true));
      pan_blocks_->push_back(CSPRep(d * 2, d, 3));
    }

    // Query selection heads + decoder.
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
    auto feats = backbone_->forward_features(images);  // {C3, C4, C5}

    // input_proj.
    std::vector<torch::Tensor> proj;
    for (int i = 0; i < cfg_.num_levels; ++i) {
      proj.push_back(input_proj_[static_cast<std::size_t>(i)]->as<nn::SequentialImpl>()->forward(
          feats[static_cast<std::size_t>(i)]));
    }

    // AIFI on the top level.
    {
      auto top = proj.back();
      const auto h = top.size(2);
      const auto w = top.size(3);
      auto src = top.flatten(2).transpose(1, 2);  // [B, HW, d]
      auto pos = SinCos2d(h, w, d, top.options());
      for (const auto& m : *aifi_) {
        src = m->as<AIFILayerImpl>()->forward(src, pos);
      }
      proj.back() = src.transpose(1, 2).reshape({b, d, h, w});
    }

    // CCFM top-down (FPN).
    std::vector<torch::Tensor> inner{proj.back()};
    for (int idx = cfg_.num_levels - 1; idx > 0; --idx) {
      const int li = cfg_.num_levels - 1 - idx;
      auto feat_high =
          lateral_convs_[static_cast<std::size_t>(li)]->as<ConvNormImpl>()->forward(inner.front());
      inner.front() = feat_high;
      auto up = F::interpolate(feat_high, F::InterpolateFuncOptions()
                                              .scale_factor(std::vector<double>{2.0, 2.0})
                                              .mode(torch::kNearest));
      auto fused = torch::cat({up, proj[static_cast<std::size_t>(idx - 1)]}, 1);
      auto inner_out = fpn_blocks_[static_cast<std::size_t>(li)]->as<CSPRepImpl>()->forward(fused);
      inner.insert(inner.begin(), inner_out);
    }

    // CCFM bottom-up (PAN).
    std::vector<torch::Tensor> outs{inner.front()};
    for (int idx = 0; idx < cfg_.num_levels - 1; ++idx) {
      auto down = downsample_convs_[static_cast<std::size_t>(idx)]->as<ConvNormImpl>()->forward(
          outs.back());
      auto fused = torch::cat({down, inner[static_cast<std::size_t>(idx + 1)]}, 1);
      outs.push_back(pan_blocks_[static_cast<std::size_t>(idx)]->as<CSPRepImpl>()->forward(fused));
    }

    // Flatten the 3 encoder feature maps into a single memory sequence.
    SpatialShapes shapes;
    std::vector<torch::Tensor> mem;
    for (const auto& o : outs) {
      shapes.emplace_back(o.size(2), o.size(3));
      mem.push_back(o.flatten(2).transpose(1, 2));  // [B, hw, d]
    }
    auto memory = torch::cat(mem, 1);  // [B, sum_hw, d]

    // Anchors + query selection.
    auto anchors = GenerateAnchors(shapes, images.options());  // [1, sum_hw, 4] (inv-sigmoid)
    auto out_mem = enc_output_norm_->forward(enc_output_->forward(memory));
    auto enc_class = enc_score_head_->forward(out_mem);           // [B, S, C]
    auto enc_coord = enc_bbox_head_->forward(out_mem) + anchors;  // [B, S, 4]

    auto topk = std::get<1>(std::get<0>(enc_class.max(-1)).topk(cfg_.num_queries, 1));  // [B, nq]
    auto gather_idx = topk.unsqueeze(-1);
    auto ref_unact = enc_coord.gather(1, gather_idx.expand({b, cfg_.num_queries, 4})).detach();
    auto target = out_mem.gather(1, gather_idx.expand({b, cfg_.num_queries, d})).detach();

    // Decoder with iterative box refinement.
    auto ref = ref_unact.sigmoid();
    torch::Tensor logits;
    torch::Tensor boxes;
    auto tgt = target;
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
    det.logits = logits;  // [B, nq, num_classes] (sigmoid/focal)
    det.boxes = boxes;    // [B, nq, 4] cxcywh
    return det;
  }

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = cfg_.name;
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.focal = true;
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/lyuwenyu/RT-DETR";
    return m;
  }

 private:
  // Grid-center anchors in inverse-sigmoid space: [1, sum_hw, 4].
  torch::Tensor GenerateAnchors(const SpatialShapes& shapes, const torch::TensorOptions& opts) {
    std::vector<torch::Tensor> anchors;
    int lvl = 0;
    for (const auto& [h, w] : shapes) {
      auto gy = torch::arange(h, opts);
      auto gx = torch::arange(w, opts);
      auto grid = torch::meshgrid({gy, gx}, "ij");
      auto xy = torch::stack({grid[1], grid[0]}, -1);  // [h, w, 2] (x, y)
      auto wht = torch::tensor({static_cast<double>(w), static_cast<double>(h)}, opts);
      auto xyn = (xy.unsqueeze(0) + 0.5) / wht;  // [1, h, w, 2]
      auto wh = torch::ones_like(xyn) * (0.05 * std::pow(2.0, lvl));
      anchors.push_back(torch::cat({xyn, wh}, -1).reshape({1, h * w, 4}));
      ++lvl;
    }
    auto a = torch::cat(anchors, 1);                           // [1, sum_hw, 4]
    auto valid = ((a > 1e-2) * (a < 1 - 1e-2)).all(-1, true);  // [1, sum_hw, 1]
    a = torch::log(a / (1 - a));
    a = torch::where(valid.expand_as(a), a, torch::full_like(a, 1e9));
    return a;
  }

  Config cfg_;
  ResNet backbone_{nullptr};
  nn::ModuleList input_proj_{nullptr};
  nn::ModuleList aifi_{nullptr};
  nn::ModuleList lateral_convs_{nullptr};
  nn::ModuleList fpn_blocks_{nullptr};
  nn::ModuleList downsample_convs_{nullptr};
  nn::ModuleList pan_blocks_{nullptr};
  nn::Linear enc_output_{nullptr};
  nn::LayerNorm enc_output_norm_{nullptr};
  nn::Linear enc_score_head_{nullptr};
  Mlp enc_bbox_head_{nullptr};
  Mlp query_pos_head_{nullptr};
  nn::ModuleList decoder_{nullptr};
  nn::ModuleList dec_score_{nullptr};
  nn::ModuleList dec_bbox_{nullptr};
};

// Sizes (n/s/m/l/x) = backbone depth + width. RT-DETR officially ships s/m/l/x
// (R18/R34/R50/R101); n is our smaller nano (R18 @ width 128).
struct SizeSpec {
  const char* tag;
  const char* backbone;
  int hidden;
};
constexpr SizeSpec kSizes[] = {
    {"n", "r18", 128}, {"s", "r18", 256}, {"m", "r34", 256}, {"l", "r50", 256}, {"x", "r101", 256},
};
// v1/v2/v3 share this inference architecture; v2/v3's published gains are largely
// training recipes (discrete sampling, dense supervision) — tracked follow-ups.
constexpr const char* kVersions[] = {"rt-detr", "rt-detrv2", "rt-detrv3"};

void RegisterOne(const std::string& name, const std::string& backbone, int hidden) {
  auto build = [name, backbone, hidden](const YAML::Node& cfg) -> std::shared_ptr<IModel> {
    Config c = ReadConfig(cfg);
    if (!(cfg && cfg["backbone"])) {
      c.backbone = backbone;
    }
    if (!(cfg && cfg["hidden_dim"])) {
      c.hidden_dim = hidden;
    }
    c.name = name;
    return std::make_shared<RtDetrImpl>(c);
  };
  ModelMeta meta;
  meta.name = name;
  meta.num_classes = 80;
  meta.num_queries = 300;
  meta.focal = true;
  meta.license = "Apache-2.0";
  meta.upstream = "https://github.com/lyuwenyu/RT-DETR";
  Registry::Instance().Register(name, meta, std::move(build));
}

}  // namespace

void RegisterRtDetr() {
  for (const char* ver : kVersions) {
    for (const auto& sz : kSizes) {
      RegisterOne(std::string(ver) + "-" + sz.tag, sz.backbone, sz.hidden);
    }
    RegisterOne(ver, "r50", 256);  // plain version name defaults to the -l config
  }
}

}  // namespace detr::models
