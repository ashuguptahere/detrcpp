// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/rt_detr.hpp"

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
namespace F = torch::nn::functional;

struct Config {
  std::string backbone{"r50"};  // r18 / r34 / r50 / r101
  std::string name{"rt-detr"};  // registry name (rt-detr[vN]-{size})
  int hidden_dim{256};          // decoder d_model
  int enc_dim{0};               // HybridEncoder/CCFM width (0 => hidden_dim; 384 for r101)
  int enc_ffn{0};               // AIFI feed-forward (0 => dim_feedforward; 2048 for r101)
  double hidden_expansion{1.0};  // CSPRepLayer hidden = out * expansion (0.5 for r18/r34)
  int nheads{8};
  int enc_layers{1};  // AIFI layers (on the top level only)
  int dec_layers{6};
  int dim_feedforward{1024};  // decoder FFN
  int num_queries{300};
  int num_classes{80};
  int num_levels{3};
  int num_points{4};
  int imgsz{640};
  int dense_o2m_k{0};        // RT-DETRv3 hierarchical dense supervision (0 = off)
  bool discrete_sample{false};  // RT-DETRv2 round-to-nearest deformable sampling
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
  x.enc_dim = Get(c, "enc_dim", x.enc_dim);
  x.enc_ffn = Get(c, "enc_ffn", x.enc_ffn);
  x.hidden_expansion = Get(c, "hidden_expansion", x.hidden_expansion);
  x.nheads = Get(c, "nheads", x.nheads);
  x.enc_layers = Get(c, "enc_layers", x.enc_layers);
  x.dec_layers = Get(c, "dec_layers", x.dec_layers);
  x.dim_feedforward = Get(c, "dim_feedforward", x.dim_feedforward);
  x.num_queries = Get(c, "num_queries", x.num_queries);
  x.num_classes = Get(c, "num_classes", x.num_classes);
  x.num_levels = Get(c, "num_levels", x.num_levels);
  x.num_points = Get(c, "num_points", x.num_points);
  x.imgsz = Get(c, "imgsz", x.imgsz);
  x.dense_o2m_k = Get(c, "dense_o2m_k", x.dense_o2m_k);
  x.discrete_sample = Get(c, "discrete_sample", x.discrete_sample);
  return x;
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

// CSPRepLayer: two 1x1 branches to a hidden width (= out * hidden_expansion), RepVGG
// bottlenecks on one, summed, then a fuse conv (conv3) back to out_channels. When
// hidden == out (expansion 1.0, RT-DETR R50/R101) conv3 is Identity and omitted — no
// conv3 weights exist in those checkpoints; R18/R34 use expansion 0.5 with a conv3.
struct CSPRepImpl : nn::Module {
  ConvNorm conv1{nullptr};
  ConvNorm conv2{nullptr};
  nn::Sequential bottlenecks{nullptr};
  ConvNorm conv3{nullptr};  // only when hidden != out
  CSPRepImpl(int in, int out, int num_blocks, double expansion) {
    const int hidden = static_cast<int>(static_cast<double>(out) * expansion);
    conv1 = register_module("conv1", ConvNorm(in, hidden, 1, 1, true));
    conv2 = register_module("conv2", ConvNorm(in, hidden, 1, 1, true));
    bottlenecks = nn::Sequential();
    for (int i = 0; i < num_blocks; ++i) {
      bottlenecks->push_back(RepVgg(hidden));
    }
    register_module("bottlenecks", bottlenecks);
    if (hidden != out) {
      conv3 = register_module("conv3", ConvNorm(hidden, out, 1, 1, true));
    }
  }
  torch::Tensor forward(torch::Tensor x) {
    auto y = bottlenecks->forward(conv1->forward(x)) + conv2->forward(x);
    return conv3.is_empty() ? y : conv3->forward(y);
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
    auto ff = linear2->forward(torch::gelu(linear1->forward(src)));  // RT-DETR AIFI uses GELU
    return norm2->forward(src + ff);
  }
};
TORCH_MODULE(AIFILayer);

class RtDetrImpl : public IModel {
 public:
  explicit RtDetrImpl(Config cfg, bool denoising = false) : cfg_(cfg), denoising_(denoising) {
    const int d = cfg.hidden_dim;                                   // decoder d_model
    const int ed = cfg.enc_dim > 0 ? cfg.enc_dim : cfg.hidden_dim;  // HybridEncoder width
    const int eff = cfg.enc_ffn > 0 ? cfg.enc_ffn : cfg.dim_feedforward;  // AIFI FFN
    const double cspx = cfg.hidden_expansion;
    auto spec = BackboneFor(cfg.backbone);
    // RT-DETR uses a ResNet-D/VD backbone (deep stem + avg-pool downsample).
    backbone_ = register_module("backbone", ResNet(spec.blocks, spec.bottleneck, /*dc5=*/false,
                                                   /*deep_stem=*/true, /*avg_down=*/true));
    const auto backbone_ch = backbone_->feature_channels();  // {C3, C4, C5}

    // input_proj (HF encoder_input_proj): 1x1 conv + BN, per backbone level -> enc dim.
    input_proj_ = register_module("input_proj", nn::ModuleList());
    for (int i = 0; i < cfg.num_levels; ++i) {
      input_proj_->push_back(nn::Sequential(
          nn::Conv2d(nn::Conv2dOptions(backbone_ch[static_cast<std::size_t>(i)], ed, 1).bias(false)),
          nn::BatchNorm2d(ed)));
    }
    // decoder_input_proj (HF): 1x1 conv + BN re-projecting each PAN output (enc dim ->
    // decoder d) before they form the deformable decoder's multi-scale memory.
    decoder_input_proj_ = register_module("decoder_input_proj", nn::ModuleList());
    for (int i = 0; i < cfg.num_levels; ++i) {
      decoder_input_proj_->push_back(
          nn::Sequential(nn::Conv2d(nn::Conv2dOptions(ed, d, 1).bias(false)), nn::BatchNorm2d(d)));
    }

    // AIFI on the top level (at the encoder width / FFN).
    aifi_ = register_module("aifi", nn::ModuleList());
    for (int i = 0; i < cfg.enc_layers; ++i) {
      aifi_->push_back(AIFILayer(ed, cfg.nheads, eff));
    }

    // CCFM top-down (FPN) + bottom-up (PAN), all at the encoder width.
    lateral_convs_ = register_module("lateral_convs", nn::ModuleList());
    fpn_blocks_ = register_module("fpn_blocks", nn::ModuleList());
    for (int i = 0; i < cfg.num_levels - 1; ++i) {
      lateral_convs_->push_back(ConvNorm(ed, ed, 1, 1, true));
      fpn_blocks_->push_back(CSPRep(ed * 2, ed, 3, cspx));
    }
    downsample_convs_ = register_module("downsample_convs", nn::ModuleList());
    pan_blocks_ = register_module("pan_blocks", nn::ModuleList());
    for (int i = 0; i < cfg.num_levels - 1; ++i) {
      downsample_convs_->push_back(ConvNorm(ed, ed, 3, 2, true));
      pan_blocks_->push_back(CSPRep(ed * 2, ed, 3, cspx));
    }

    // Shared query selection + deformable decoder.
    head_ = BuildDeformDetectHead(*this, d, cfg.num_levels, cfg.nheads, cfg.num_points,
                                  cfg.dim_feedforward, cfg.dec_layers, cfg.num_classes,
                                  cfg.num_queries, cfg.discrete_sample);
    if (denoising_) {
      // RT-DETR-CDN: content embedding for denoising queries (+1 unused row).
      // Registered only for rt-detr-cdn, so plain rt-detr stays byte-identical.
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
    DeformCdn cdn;
    cdn.active = true;
    cdn.num_dn = dn_in.num_dn;
    cdn.dn_tgt = label_enc_->forward(dn_in.dn_labels.transpose(0, 1).contiguous());  // [B,num_dn,d]
    cdn.dn_ref = dn_in.dn_ref.transpose(0, 1).contiguous();                          // [B,num_dn,4]
    cdn.attn_mask = dn_in.attn_mask;
    return RunDeformDetectHead(head_, enc.memory, enc.shapes, cdn, dn_out);
  }

 private:
  struct Encoded {
    torch::Tensor memory;  // [B, Sum(HW), d]
    SpatialShapes shapes;
  };

  // Backbone + input_proj + AIFI + CCFM (FPN/PAN) + decoder_input_proj -> the
  // flattened multi-scale memory the deformable head consumes (shared forward).
  Encoded Encode(torch::Tensor images) {
    const int ed = cfg_.enc_dim > 0 ? cfg_.enc_dim : cfg_.hidden_dim;  // HybridEncoder width
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
      auto src = top.flatten(2).transpose(1, 2);  // [B, HW, ed]
      auto pos = SinCos2d(h, w, ed, top.options());
      for (const auto& m : *aifi_) {
        src = m->as<AIFILayerImpl>()->forward(src, pos);
      }
      proj.back() = src.transpose(1, 2).reshape({b, ed, h, w});
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

    // decoder_input_proj, then flatten the 3 maps into one memory sequence.
    SpatialShapes shapes;
    std::vector<torch::Tensor> mem;
    for (std::size_t i = 0; i < outs.size(); ++i) {
      auto o = decoder_input_proj_[i]->as<nn::SequentialImpl>()->forward(outs[i]);
      shapes.emplace_back(o.size(2), o.size(3));
      mem.push_back(o.flatten(2).transpose(1, 2));  // [B, hw, d]
    }
    return {torch::cat(mem, 1), shapes};
  }

 public:

  ModelMeta Meta() const override {
    ModelMeta m;
    m.name = denoising_ ? "rt-detr-cdn" : cfg_.name;
    m.imgsz = cfg_.imgsz;
    m.num_classes = cfg_.num_classes;
    m.num_queries = cfg_.num_queries;
    m.focal = true;
    m.imagenet_norm = false;  // RT-DETR trains on raw [0,1], square resize
    m.license = "Apache-2.0";
    m.upstream = "https://github.com/lyuwenyu/RT-DETR";
    return m;
  }

  // RT-DETRv3: one-to-many dense positive supervision (k>0 only for the v3 matrix).
  int DenseSupervisionK() const override { return cfg_.dense_o2m_k; }

 private:
  Config cfg_;
  bool denoising_{false};
  ResNet backbone_{nullptr};
  nn::ModuleList input_proj_{nullptr};
  nn::ModuleList decoder_input_proj_{nullptr};
  nn::ModuleList aifi_{nullptr};
  nn::ModuleList lateral_convs_{nullptr};
  nn::ModuleList fpn_blocks_{nullptr};
  nn::ModuleList downsample_convs_{nullptr};
  nn::ModuleList pan_blocks_{nullptr};
  DeformDetectHead head_;
  nn::Embedding label_enc_{nullptr};
};

// Sizes (n/s/m/l/x) scale the whole stack to match the official RT-DETR checkpoints
// (R18/R34/R50/R101): backbone depth + decoder depth + HybridEncoder width/FFN +
// CCFM CSPRepLayer expansion. R18/R34 run a 256-wide encoder with a 0.5 CSP expansion
// (half-width RepVGG bottlenecks + a conv3 fuse) and 3/4 decoder layers; R50 is the
// reference 256/1.0/6; R101 widens the encoder to 384 (FFN 2048). n is our smaller
// nano (R18 @ width 128, no official weights).
struct SizeSpec {
  const char* tag;
  const char* backbone;
  int hidden;        // decoder d_model
  int dec_layers;
  int enc_dim;       // HybridEncoder/CCFM width
  int enc_ffn;       // AIFI feed-forward
  double expansion;  // CSPRepLayer hidden_expansion
};
constexpr SizeSpec kSizes[] = {
    {"n", "r18", 128, 3, 128, 512, 0.5},  {"s", "r18", 256, 3, 256, 1024, 0.5},
    {"m", "r34", 256, 4, 256, 1024, 0.5}, {"l", "r50", 256, 6, 256, 1024, 1.0},
    {"x", "r101", 256, 6, 384, 2048, 1.0},
};
// v1/v2/v3 share this inference architecture; v3's published gain is largely a
// training recipe — hierarchical dense positive supervision (one-to-many matching),
// enabled here via dense_o2m_k. (v2's discrete sampling is a tracked follow-up.)
constexpr const char* kVersions[] = {"rt-detr", "rt-detrv2", "rt-detrv3"};
constexpr int kDenseV3 = 6;  // RT-DETRv3 one-to-many top-k per GT

void RegisterOne(const std::string& name, const SizeSpec& sz, int dense_k, bool discrete) {
  auto build = [name, sz, dense_k, discrete](const YAML::Node& cfg) -> std::shared_ptr<IModel> {
    Config c = ReadConfig(cfg);
    if (!(cfg && cfg["backbone"])) {
      c.backbone = sz.backbone;
    }
    if (!(cfg && cfg["hidden_dim"])) {
      c.hidden_dim = sz.hidden;
    }
    if (!(cfg && cfg["dec_layers"])) {
      c.dec_layers = sz.dec_layers;
    }
    if (!(cfg && cfg["enc_dim"])) {
      c.enc_dim = sz.enc_dim;
    }
    if (!(cfg && cfg["enc_ffn"])) {
      c.enc_ffn = sz.enc_ffn;
    }
    if (!(cfg && cfg["hidden_expansion"])) {
      c.hidden_expansion = sz.expansion;
    }
    if (!(cfg && cfg["dense_o2m_k"])) {
      c.dense_o2m_k = dense_k;
    }
    if (!(cfg && cfg["discrete_sample"])) {
      c.discrete_sample = discrete;
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
    const int dense_k = (std::string(ver) == "rt-detrv3") ? kDenseV3 : 0;
    const bool discrete = (std::string(ver) == "rt-detrv2");  // discrete sampling = v2's
    for (const auto& sz : kSizes) {
      RegisterOne(std::string(ver) + "-" + sz.tag, sz, dense_k, discrete);
    }
    // Plain version = the -l (R50) config.
    RegisterOne(ver, {"", "r50", 256, 6, 256, 1024, 1.0}, dense_k, discrete);
  }

  // RT-DETR-CDN: RT-DETR (the -l config) + contrastive denoising training. Like
  // the other -cdn variants, a single training-recipe alias (not a size matrix).
  auto cdn_build = [](const YAML::Node& cfg) -> std::shared_ptr<IModel> {
    Config c = ReadConfig(cfg);
    if (!(cfg && cfg["backbone"])) {
      c.backbone = "r50";
    }
    if (!(cfg && cfg["hidden_dim"])) {
      c.hidden_dim = 256;
    }
    c.name = "rt-detr-cdn";
    return std::make_shared<RtDetrImpl>(c, /*denoising=*/true);
  };
  ModelMeta cdn_meta;
  cdn_meta.name = "rt-detr-cdn";
  cdn_meta.num_classes = 80;
  cdn_meta.num_queries = 300;
  cdn_meta.focal = true;
  cdn_meta.imagenet_norm = false;
  cdn_meta.license = "Apache-2.0";
  cdn_meta.upstream = "https://github.com/lyuwenyu/RT-DETR";
  Registry::Instance().Register("rt-detr-cdn", cdn_meta, std::move(cdn_build));
}

}  // namespace detr::models
