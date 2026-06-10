// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Mirrors models::DetrImpl::Forward op-for-op as a fixed-shape (batch 1) ONNX
// graph. Every shape is static (imgsz is fixed at export), so reshapes use
// constant targets and there are no dynamic-shape ops.

#include "detr/onnxexport/detr_export.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "detr/onnxexport/graph_builder.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/tensor.hpp"

namespace detr::onnxexport {

namespace {

using weights::DType;
using weights::RawTensor;

constexpr float kBnEps = 1e-5F;
constexpr float kLnEps = 1e-5F;

struct Emitter {
  GraphBuilder& g;
  const weights::StateDict& sd;
  DetrArch a;
  std::optional<core::Error> error;

  void Fail(std::string msg) {
    if (!error) {
      error = core::Error{core::ErrorCode::NotFound, std::move(msg)};
    }
  }

  std::vector<float> Floats(const RawTensor& t) {
    std::vector<float> v(t.data.size() / sizeof(float));
    if (!v.empty()) {
      std::memcpy(v.data(), t.data.data(), t.data.size());
    }
    return v;
  }

  // Adds the StateDict tensor |name| as an initializer of the same name.
  const RawTensor* Need(const std::string& name) {
    const RawTensor* t = sd.Find(name);
    if (t == nullptr) {
      Fail(fmt::format("missing weight '{}'", name));
      return nullptr;
    }
    if (t->dtype != DType::F32) {
      Fail(fmt::format("weight '{}' is not f32", name));
      return nullptr;
    }
    return t;
  }

  void AddWeight(const std::string& name) {
    const RawTensor* t = Need(name);
    if (t != nullptr) {
      g.AddInitializerF32(name, t->shape, Floats(*t));
    }
  }

  void AddFloatConst(const std::string& name, const std::vector<std::int64_t>& dims,
                     const std::vector<float>& data) {
    g.AddInitializerF32(name, dims, data);
  }

  void AddShape(const std::string& name, const std::vector<std::int64_t>& dims) {
    g.AddInitializerI64(name, {static_cast<std::int64_t>(dims.size())}, dims);
  }

  // --- elementary nodes ---
  std::string Conv(const std::string& x, const std::string& w, const std::string& bias, int k,
                   int stride, int pad) {
    AddWeight(w);
    std::vector<std::string> ins{x, w};
    if (!bias.empty()) {
      AddWeight(bias);
      ins.push_back(bias);
    }
    std::string y = g.Unique("conv");
    g.AddNode("Conv", ins, {y});
    g.AttrInts("kernel_shape", {k, k});
    g.AttrInts("strides", {stride, stride});
    g.AttrInts("pads", {pad, pad, pad, pad});
    g.AttrInt("group", 1);
    return y;
  }

  std::string Bn(const std::string& x, const std::string& prefix) {
    AddWeight(prefix + ".weight");
    AddWeight(prefix + ".bias");
    AddWeight(prefix + ".running_mean");
    AddWeight(prefix + ".running_var");
    std::string y = g.Unique("bn");
    g.AddNode("BatchNormalization",
              {x, prefix + ".weight", prefix + ".bias", prefix + ".running_mean",
               prefix + ".running_var"},
              {y});
    g.AttrFloat("epsilon", kBnEps);
    return y;
  }

  std::string Unary(const std::string& op, const std::string& x) {
    std::string y = g.Unique(op);
    g.AddNode(op, {x}, {y});
    return y;
  }

  std::string Pool(const std::string& x, int k, int stride, int pad) {
    std::string y = g.Unique("pool");
    g.AddNode("MaxPool", {x}, {y});
    g.AttrInts("kernel_shape", {k, k});
    g.AttrInts("strides", {stride, stride});
    g.AttrInts("pads", {pad, pad, pad, pad});
    return y;
  }

  std::string Binary(const std::string& op, const std::string& x, const std::string& y) {
    std::string z = g.Unique(op);
    g.AddNode(op, {x, y}, {z});
    return z;
  }

  // Y = X @ W^T + B   (linear).  weight [out,in], bias [out].
  std::string Gemm(const std::string& x, const std::string& w, const std::string& b) {
    AddWeight(w);
    AddWeight(b);
    std::string y = g.Unique("gemm");
    g.AddNode("Gemm", {x, w, b}, {y});
    g.AttrInt("transB", 1);
    return y;
  }

  std::string LayerNorm(const std::string& x, const std::string& prefix) {
    AddWeight(prefix + ".weight");
    AddWeight(prefix + ".bias");
    std::string y = g.Unique("ln");
    g.AddNode("LayerNormalization", {x, prefix + ".weight", prefix + ".bias"}, {y});
    g.AttrInt("axis", -1);
    g.AttrFloat("epsilon", kLnEps);
    return y;
  }

  std::string Reshape(const std::string& x, const std::vector<std::int64_t>& dims) {
    std::string sh = g.Unique("shape");
    AddShape(sh, dims);
    std::string y = g.Unique("reshape");
    g.AddNode("Reshape", {x, sh}, {y});
    return y;
  }

  std::string Transpose(const std::string& x, const std::vector<std::int64_t>& perm) {
    std::string y = g.Unique("transpose");
    g.AddNode("Transpose", {x}, {y});
    g.AttrInts("perm", perm);
    return y;
  }

  std::string Softmax(const std::string& x, int axis) {
    std::string y = g.Unique("softmax");
    g.AddNode("Softmax", {x}, {y});
    g.AttrInt("axis", axis);
    return y;
  }

  // Slices rows [r0,r1) of a [R,C] weight into a new [r1-r0, C] initializer.
  void AddRowSlice(const std::string& dst, const RawTensor* src, int r0, int r1) {
    if (src == nullptr) {
      return;
    }
    const auto cols = static_cast<int>(src->shape.size() == 2 ? src->shape[1] : 0);
    const auto all = Floats(*src);
    std::vector<float> sub(all.begin() + static_cast<long>(r0 * cols),
                           all.begin() + static_cast<long>(r1 * cols));
    AddFloatConst(dst, {r1 - r0, cols}, sub);
  }
  void AddVecSlice(const std::string& dst, const RawTensor* src, int i0, int i1) {
    if (src == nullptr) {
      return;
    }
    const auto all = Floats(*src);
    std::vector<float> sub(all.begin() + i0, all.begin() + i1);
    AddFloatConst(dst, {i1 - i0}, sub);
  }

  // Multi-head attention, decomposed. Returns the [Lq, d] output.
  std::string Mha(const std::string& qin, const std::string& kin, const std::string& vin,
                  const std::string& prefix, int lq, int lk) {
    const int d = a.hidden_dim;
    const int nh = a.nheads;
    const int hd = d / nh;

    const RawTensor* ipw = Need(prefix + ".in_proj_weight");  // [3d, d]
    const RawTensor* ipb = Need(prefix + ".in_proj_bias");    // [3d]
    AddRowSlice(prefix + ".Wq", ipw, 0, d);
    AddRowSlice(prefix + ".Wk", ipw, d, 2 * d);
    AddRowSlice(prefix + ".Wv", ipw, 2 * d, 3 * d);
    AddVecSlice(prefix + ".bq", ipb, 0, d);
    AddVecSlice(prefix + ".bk", ipb, d, 2 * d);
    AddVecSlice(prefix + ".bv", ipb, 2 * d, 3 * d);

    // Projections.  Wq/Wk/Wv already added as initializers; Gemm() would re-add,
    // so call AddNode directly using the existing initializer names.
    auto proj = [&](const std::string& x, const std::string& w, const std::string& b) {
      std::string y = g.Unique("gemm");
      g.AddNode("Gemm", {x, w, b}, {y});
      g.AttrInt("transB", 1);
      return y;
    };
    std::string qp = proj(qin, prefix + ".Wq", prefix + ".bq");  // [lq,d]
    std::string kp = proj(kin, prefix + ".Wk", prefix + ".bk");  // [lk,d]
    std::string vp = proj(vin, prefix + ".Wv", prefix + ".bv");  // [lk,d]

    // scale q by 1/sqrt(hd)
    const std::string scale = g.Unique("scale");
    AddFloatConst(scale, {1}, {1.0F / std::sqrt(static_cast<float>(hd))});
    std::string qs = Binary("Mul", qp, scale);

    // to heads: [L,d] -> [L,nh,hd] -> [nh,L,hd]
    std::string qh = Transpose(Reshape(qs, {lq, nh, hd}), {1, 0, 2});
    std::string kh = Transpose(Reshape(kp, {lk, nh, hd}), {1, 0, 2});
    std::string vh = Transpose(Reshape(vp, {lk, nh, hd}), {1, 0, 2});

    std::string khT = Transpose(kh, {0, 2, 1});      // [nh,hd,lk]
    std::string scores = Binary("MatMul", qh, khT);  // [nh,lq,lk]
    std::string attn = Softmax(scores, -1);
    std::string ctx = Binary("MatMul", attn, vh);  // [nh,lq,hd]

    std::string back = Transpose(ctx, {1, 0, 2});  // [lq,nh,hd]
    std::string flat = Reshape(back, {lq, d});     // [lq,d]
    return Gemm(flat, prefix + ".out_proj.weight", prefix + ".out_proj.bias");
  }

  std::string Concat(const std::vector<std::string>& ins, int axis) {
    std::string y = g.Unique("concat");
    g.AddNode("Concat", ins, {y});
    g.AttrInt("axis", axis);
    return y;
  }

  // Conditional/DAB decoupled attention: q/k/v are ALREADY projected, with
  // separate query/key width |qk_dim| and value width |v_dim|. Returns the
  // context [lq, v_dim] (the caller applies the out projection).
  std::string DecoupledMha(const std::string& q, const std::string& k, const std::string& v, int lq,
                           int lk, int qk_dim, int v_dim) {
    const int nh = a.nheads;
    const int qk_hd = qk_dim / nh;
    const int v_hd = v_dim / nh;
    const std::string scale = g.Unique("scale");
    AddFloatConst(scale, {1}, {1.0F / std::sqrt(static_cast<float>(qk_hd))});
    std::string qs = Binary("Mul", q, scale);
    std::string qh = Transpose(Reshape(qs, {lq, nh, qk_hd}), {1, 0, 2});  // [nh,lq,qk_hd]
    std::string kh = Transpose(Reshape(k, {lk, nh, qk_hd}), {1, 0, 2});
    std::string vh = Transpose(Reshape(v, {lk, nh, v_hd}), {1, 0, 2});
    std::string scores = Binary("MatMul", qh, Transpose(kh, {0, 2, 1}));  // [nh,lq,lk]
    std::string ctx = Binary("MatMul", Softmax(scores, -1), vh);          // [nh,lq,v_hd]
    return Reshape(Transpose(ctx, {1, 0, 2}), {lq, v_dim});               // [lq,v_dim]
  }

  // Conditional-DETR's reference is a fixed function of the learned query
  // embeddings, so the per-query sine embedding and inverse-sigmoid reference
  // padding are constants — precompute them exactly (matching SineEmbedForRef).
  struct CondConsts {
    std::vector<float> query_sine_base;  // [nq, d]
    std::vector<float> ref_pad;          // [nq, 4]
  };
  CondConsts CondPrecompute() {
    const int d = a.hidden_dim;
    const int half = d / 2;
    const int nq = a.num_queries;
    CondConsts c;
    const RawTensor* qe = Need("query_embed.weight");              // [nq,d]
    const RawTensor* w0 = Need("ref_point_head.layers.0.weight");  // [d,d]
    const RawTensor* b0 = Need("ref_point_head.layers.0.bias");    // [d]
    const RawTensor* w1 = Need("ref_point_head.layers.1.weight");  // [2,d]
    const RawTensor* b1 = Need("ref_point_head.layers.1.bias");    // [2]
    if (qe == nullptr || w0 == nullptr || b0 == nullptr || w1 == nullptr || b1 == nullptr) {
      return c;
    }
    const auto QE = Floats(*qe), W0 = Floats(*w0), B0 = Floats(*b0), W1 = Floats(*w1),
               B1 = Floats(*b1);
    const double scale = 2.0 * std::numbers::pi;
    std::vector<double> dim_t(static_cast<std::size_t>(half));
    for (int k = 0; k < half; ++k) {
      dim_t[static_cast<std::size_t>(k)] =
          std::pow(10000.0, 2.0 * std::floor(k / 2.0) / static_cast<double>(half));
    }
    auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    auto inv_sig = [](double s) {
      const double eps = 1e-5;
      return std::log(std::max(s, eps) / std::max(1.0 - s, eps));
    };
    c.query_sine_base.assign(static_cast<std::size_t>(nq) * static_cast<std::size_t>(d), 0.0F);
    c.ref_pad.assign(static_cast<std::size_t>(nq) * 4, 0.0F);
    for (int q = 0; q < nq; ++q) {
      std::vector<double> hh(static_cast<std::size_t>(d));
      for (int o = 0; o < d; ++o) {
        double s = B0[static_cast<std::size_t>(o)];
        for (int i = 0; i < d; ++i) {
          s += static_cast<double>(W0[static_cast<std::size_t>(o * d + i)]) *
               static_cast<double>(QE[static_cast<std::size_t>(q * d + i)]);
        }
        hh[static_cast<std::size_t>(o)] = std::max(0.0, s);
      }
      double r[2];
      for (int o = 0; o < 2; ++o) {
        double s = B1[static_cast<std::size_t>(o)];
        for (int i = 0; i < d; ++i) {
          s += static_cast<double>(W1[static_cast<std::size_t>(o * d + i)]) *
               hh[static_cast<std::size_t>(i)];
        }
        r[o] = s;
      }
      const double refx = sig(r[0]);
      const double refy = sig(r[1]);
      const auto base = static_cast<std::size_t>(q) * static_cast<std::size_t>(d);
      for (int j = 0; j < half; j += 2) {
        const double yp0 = refy * scale / dim_t[static_cast<std::size_t>(j)];
        const double yp1 = refy * scale / dim_t[static_cast<std::size_t>(j + 1)];
        const double xp0 = refx * scale / dim_t[static_cast<std::size_t>(j)];
        const double xp1 = refx * scale / dim_t[static_cast<std::size_t>(j + 1)];
        c.query_sine_base[base + static_cast<std::size_t>(j)] = static_cast<float>(std::sin(yp0));
        c.query_sine_base[base + static_cast<std::size_t>(j + 1)] = static_cast<float>(std::cos(yp1));
        c.query_sine_base[base + static_cast<std::size_t>(half + j)] =
            static_cast<float>(std::sin(xp0));
        c.query_sine_base[base + static_cast<std::size_t>(half + j + 1)] =
            static_cast<float>(std::cos(xp1));
      }
      const auto rb = static_cast<std::size_t>(q) * 4;
      c.ref_pad[rb + 0] = static_cast<float>(inv_sig(refx));
      c.ref_pad[rb + 1] = static_cast<float>(inv_sig(refy));
    }
    return c;
  }

  // One conditional decoder layer: decoupled self-attention, conditional
  // cross-attention (query content + sine concatenated head-wise), FFN.
  std::string CondLayer(const std::string& p, std::string tgt, const std::string& memory,
                        const std::string& pos, const std::string& query_pos,
                        const std::string& query_sine, bool is_first, int Q, int L) {
    const int d = a.hidden_dim;
    const int nh = a.nheads;
    const int hd = d / nh;
    // Decoupled self-attention.
    std::string q = Binary("Add", Gemm(tgt, p + ".sa_qcontent_proj.weight", p + ".sa_qcontent_proj.bias"),
                           Gemm(query_pos, p + ".sa_qpos_proj.weight", p + ".sa_qpos_proj.bias"));
    std::string k = Binary("Add", Gemm(tgt, p + ".sa_kcontent_proj.weight", p + ".sa_kcontent_proj.bias"),
                           Gemm(query_pos, p + ".sa_kpos_proj.weight", p + ".sa_kpos_proj.bias"));
    std::string v = Gemm(tgt, p + ".sa_v_proj.weight", p + ".sa_v_proj.bias");
    std::string sa = Gemm(DecoupledMha(q, k, v, Q, Q, d, d), p + ".sa_out_proj.weight",
                          p + ".sa_out_proj.bias");
    tgt = LayerNorm(Binary("Add", tgt, sa), p + ".norm1");
    // Conditional cross-attention.
    std::string qc = Gemm(tgt, p + ".ca_qcontent_proj.weight", p + ".ca_qcontent_proj.bias");
    std::string kc = Gemm(memory, p + ".ca_kcontent_proj.weight", p + ".ca_kcontent_proj.bias");
    std::string vv = Gemm(memory, p + ".ca_v_proj.weight", p + ".ca_v_proj.bias");
    std::string kp = Gemm(pos, p + ".ca_kpos_proj.weight", p + ".ca_kpos_proj.bias");
    std::string qq = qc;
    std::string kk = kc;
    if (is_first) {
      qq = Binary("Add", qc, Gemm(query_pos, p + ".ca_qpos_proj.weight", p + ".ca_qpos_proj.bias"));
      kk = Binary("Add", kc, kp);
    }
    std::string qse = Gemm(query_sine, p + ".ca_qpos_sine_proj.weight", p + ".ca_qpos_sine_proj.bias");
    std::string qcat = Reshape(Concat({Reshape(qq, {Q, nh, hd}), Reshape(qse, {Q, nh, hd})}, 2),
                               {Q, 2 * d});
    std::string kcat = Reshape(Concat({Reshape(kk, {L, nh, hd}), Reshape(kp, {L, nh, hd})}, 2),
                               {L, 2 * d});
    std::string ca = Gemm(DecoupledMha(qcat, kcat, vv, Q, L, 2 * d, d), p + ".ca_out_proj.weight",
                          p + ".ca_out_proj.bias");
    tgt = LayerNorm(Binary("Add", tgt, ca), p + ".norm2");
    // FFN.
    std::string h1 = Gemm(tgt, p + ".linear1.weight", p + ".linear1.bias");
    std::string ff = Gemm(Unary("Relu", h1), p + ".linear2.weight", p + ".linear2.bias");
    return LayerNorm(Binary("Add", tgt, ff), p + ".norm3");
  }

  // --- backbones (return the final feature map [1, C, h, w]) ---

  std::string CompactBackbone(const std::string& input) {
    std::string x = Conv(input, "backbone.0.weight", "", 7, 2, 3);
    x = Bn(x, "backbone.1");
    x = Unary("Relu", x);
    x = Pool(x, 3, 2, 1);
    const int stage_base[4] = {4, 10, 16, 22};
    for (int s = 0; s < 4; ++s) {
      const int base = stage_base[s];
      const int stride = (s == 0) ? 1 : 2;
      x = Conv(x, fmt::format("backbone.{}.weight", base), "", 3, stride, 1);
      x = Bn(x, fmt::format("backbone.{}", base + 1));
      x = Unary("Relu", x);
      x = Conv(x, fmt::format("backbone.{}.weight", base + 3), "", 3, 1, 1);
      x = Bn(x, fmt::format("backbone.{}", base + 4));
      x = Unary("Relu", x);
    }
    return x;
  }

  // ResNet-50 bottleneck: 1x1 -> 3x3(stride) -> 1x1, plus an optional 1x1
  // downsample on the residual; relu(main + identity).
  std::string Bottleneck(const std::string& x, const std::string& p, int stride, bool has_down) {
    std::string c = Unary("Relu", Bn(Conv(x, p + ".conv1.weight", "", 1, 1, 0), p + ".bn1"));
    c = Unary("Relu", Bn(Conv(c, p + ".conv2.weight", "", 3, stride, 1), p + ".bn2"));
    c = Bn(Conv(c, p + ".conv3.weight", "", 1, 1, 0), p + ".bn3");
    std::string id = x;
    if (has_down) {
      id = Bn(Conv(x, p + ".downsample.0.weight", "", 1, stride, 0), p + ".downsample.1");
    }
    return Unary("Relu", Binary("Add", c, id));
  }

  std::string ResNet50Backbone(const std::string& input) {
    std::string x =
        Unary("Relu", Bn(Conv(input, "backbone.conv1.weight", "", 7, 2, 3), "backbone.bn1"));
    x = Pool(x, 3, 2, 1);
    const auto& blocks = a.resnet_blocks;
    const int strides[4] = {1, 2, 2, 2};
    for (int lyr = 0; lyr < 4; ++lyr) {
      for (int b = 0; b < blocks[static_cast<std::size_t>(lyr)]; ++b) {
        const std::string p = fmt::format("backbone.layer{}.{}", lyr + 1, b);
        const int stride = (b == 0) ? strides[lyr] : 1;
        x = Bottleneck(x, p, stride, /*has_down=*/b == 0);
      }
    }
    return x;
  }

  // Replicates SinePos -> a constant [L, d] positional encoding (L = h*w).
  std::vector<float> SinePos(int h, int w) {
    const int d = a.hidden_dim;
    const int half = d / 2;
    const double scale = 2.0 * std::numbers::pi;
    std::vector<double> dim_t(static_cast<std::size_t>(half));
    for (int k = 0; k < half; ++k) {
      dim_t[static_cast<std::size_t>(k)] =
          std::pow(10000.0, 2.0 * std::floor(k / 2.0) / static_cast<double>(half));
    }
    std::vector<float> pos(static_cast<std::size_t>(h) * static_cast<std::size_t>(w) *
                           static_cast<std::size_t>(d));
    for (int i = 0; i < h; ++i) {
      const double ys = (i + 1) / (static_cast<double>(h) + 1e-6) * scale;
      for (int j = 0; j < w; ++j) {
        const double xs = (j + 1) / (static_cast<double>(w) + 1e-6) * scale;
        const std::size_t base = (static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
                                  static_cast<std::size_t>(j)) *
                                 static_cast<std::size_t>(d);
        for (int c = 0; c < half; ++c) {
          const double v = ys / dim_t[static_cast<std::size_t>(c)];
          pos[base + static_cast<std::size_t>(c)] =
              static_cast<float>((c % 2 == 0) ? std::sin(v) : std::cos(v));
        }
        for (int c = 0; c < half; ++c) {
          const double v = xs / dim_t[static_cast<std::size_t>(c)];
          pos[base + static_cast<std::size_t>(half + c)] =
              static_cast<float>((c % 2 == 0) ? std::sin(v) : std::cos(v));
        }
      }
    }
    return pos;
  }
};

}  // namespace

core::Result<void> ExportDetr(const DetrArch& arch, const weights::StateDict& weights,
                              const std::string& path) {
  GraphBuilder g("detr");
  Emitter e{g, weights, arch, std::nullopt};

  const int d = arch.hidden_dim;
  const int feat = arch.imgsz / 32;  // backbone downsamples by 32x
  const int L = feat * feat;
  const int Q = arch.num_queries;

  g.AddInput("images", {1, 3, arch.imgsz, arch.imgsz});

  // --- backbone (compact conv stack or torchvision ResNet-50) ---
  std::string backbone_out = (arch.backbone == Backbone::ResNet50) ? e.ResNet50Backbone("images")
                                                                   : e.CompactBackbone("images");

  // input_proj 1x1 conv (has bias) -> [1,d,feat,feat]. The weight initializer
  // carries the input channel count, so this is the same for both backbones.
  std::string x = e.Conv(backbone_out, "input_proj.weight", "input_proj.bias", 1, 1, 0);

  // [1,d,h,w] -> [d, L] -> [L, d]
  std::string src = e.Transpose(e.Reshape(x, {d, L}), {1, 0});

  // positional encoding constant [L, d]
  e.AddFloatConst("pos", {L, d}, e.SinePos(feat, feat));
  const std::string pos = "pos";

  // --- encoder ---
  std::string memory = src;
  for (int i = 0; i < arch.enc_layers; ++i) {
    const std::string p = fmt::format("encoder.{}", i);
    std::string q = e.Binary("Add", memory, pos);
    std::string attn = e.Mha(q, q, memory, p + ".self_attn", L, L);
    memory = e.LayerNorm(e.Binary("Add", memory, attn), p + ".norm1");
    std::string h1 = e.Gemm(memory, p + ".linear1.weight", p + ".linear1.bias");
    std::string ff = e.Gemm(e.Unary("Relu", h1), p + ".linear2.weight", p + ".linear2.bias");
    memory = e.LayerNorm(e.Binary("Add", memory, ff), p + ".norm2");
  }

  // --- decoder ---
  e.AddWeight("query_embed.weight");  // [Q, d]
  const std::string query_pos = "query_embed.weight";
  e.AddFloatConst(
      "dec_tgt0", {Q, d},
      std::vector<float>(static_cast<std::size_t>(Q) * static_cast<std::size_t>(d), 0.0F));
  std::string tgt = "dec_tgt0";
  for (int i = 0; i < arch.dec_layers; ++i) {
    const std::string p = fmt::format("decoder.{}", i);
    std::string q = e.Binary("Add", tgt, query_pos);
    std::string sa = e.Mha(q, q, tgt, p + ".self_attn", Q, Q);
    tgt = e.LayerNorm(e.Binary("Add", tgt, sa), p + ".norm1");
    std::string qc = e.Binary("Add", tgt, query_pos);
    std::string kc = e.Binary("Add", memory, pos);
    std::string ca = e.Mha(qc, kc, memory, p + ".cross_attn", Q, L);
    tgt = e.LayerNorm(e.Binary("Add", tgt, ca), p + ".norm2");
    std::string h1 = e.Gemm(tgt, p + ".linear1.weight", p + ".linear1.bias");
    std::string ff = e.Gemm(e.Unary("Relu", h1), p + ".linear2.weight", p + ".linear2.bias");
    tgt = e.LayerNorm(e.Binary("Add", tgt, ff), p + ".norm3");
  }
  tgt = e.LayerNorm(tgt, "decoder_norm");  // DETR's final decoder LayerNorm

  // --- heads ---
  std::string logits2d = e.Gemm(tgt, "class_embed.weight", "class_embed.bias");  // [Q, C+1]
  std::string b0 = e.Unary("Relu", e.Gemm(tgt, "bbox_embed.0.weight", "bbox_embed.0.bias"));
  std::string b1 = e.Unary("Relu", e.Gemm(b0, "bbox_embed.2.weight", "bbox_embed.2.bias"));
  std::string boxes2d = e.Unary("Sigmoid", e.Gemm(b1, "bbox_embed.4.weight", "bbox_embed.4.bias"));

  // [Q, *] -> [1, Q, *]
  std::string logits = e.Reshape(logits2d, {1, Q, arch.num_classes + 1});
  std::string boxes = e.Reshape(boxes2d, {1, Q, 4});
  // Rename to fixed output names by an Identity (keeps graph outputs stable).
  g.AddNode("Identity", {logits}, {"logits"});
  g.AddNode("Identity", {boxes}, {"boxes"});
  g.AddOutput("logits", {1, Q, arch.num_classes + 1});
  g.AddOutput("boxes", {1, Q, 4});

  if (e.error) {
    return tl::make_unexpected(*e.error);
  }
  return g.Save(path, /*opset=*/17);
}

core::Result<void> ExportConditional(const DetrArch& arch, const weights::StateDict& weights,
                                     const std::string& path) {
  GraphBuilder g("conditional-detr");
  Emitter e{g, weights, arch, std::nullopt};

  const int d = arch.hidden_dim;
  const int feat = arch.imgsz / 32;  // ResNet-50 downsamples by 32x
  const int L = feat * feat;
  const int Q = arch.num_queries;
  const int C = arch.num_classes;  // focal: no no-object slot

  g.AddInput("images", {1, 3, arch.imgsz, arch.imgsz});

  std::string x = e.Conv(e.ResNet50Backbone("images"), "input_proj.weight", "input_proj.bias", 1, 1, 0);
  std::string src = e.Transpose(e.Reshape(x, {d, L}), {1, 0});  // [L, d]
  e.AddFloatConst("pos", {L, d}, e.SinePos(feat, feat));
  const std::string pos = "pos";

  // --- encoder (standard self-attention) ---
  std::string memory = src;
  for (int i = 0; i < arch.enc_layers; ++i) {
    const std::string p = fmt::format("encoder.{}", i);
    std::string q = e.Binary("Add", memory, pos);
    std::string attn = e.Mha(q, q, memory, p + ".self_attn", L, L);
    memory = e.LayerNorm(e.Binary("Add", memory, attn), p + ".norm1");
    std::string h1 = e.Gemm(memory, p + ".linear1.weight", p + ".linear1.bias");
    std::string ff = e.Gemm(e.Unary("Relu", h1), p + ".linear2.weight", p + ".linear2.bias");
    memory = e.LayerNorm(e.Binary("Add", memory, ff), p + ".norm2");
  }

  // --- decoder (conditional, fixed reference) ---
  const Emitter::CondConsts cc = e.CondPrecompute();
  e.AddFloatConst("query_sine_base", {Q, d}, cc.query_sine_base);
  e.AddFloatConst("ref_pad", {Q, 4}, cc.ref_pad);
  e.AddWeight("query_embed.weight");
  const std::string query_pos = "query_embed.weight";  // [Q, d]
  e.AddFloatConst("dec_tgt0", {Q, d},
                  std::vector<float>(static_cast<std::size_t>(Q) * static_cast<std::size_t>(d), 0.0F));
  std::string tgt = "dec_tgt0";
  for (int i = 0; i < arch.dec_layers; ++i) {
    const std::string p = fmt::format("decoder.{}", i);
    std::string query_sine = "query_sine_base";
    if (i > 0) {
      std::string qs = e.Gemm(e.Unary("Relu", e.Gemm(tgt, "query_scale.layers.0.weight",
                                                     "query_scale.layers.0.bias")),
                              "query_scale.layers.1.weight", "query_scale.layers.1.bias");
      query_sine = e.Binary("Mul", "query_sine_base", qs);
    }
    tgt = e.CondLayer(p, tgt, memory, pos, query_pos, query_sine, /*is_first=*/i == 0, Q, L);
  }
  tgt = e.LayerNorm(tgt, "decoder_norm");

  // --- heads (focal: sigmoid over C classes; box = sigmoid(mlp + ref_pad)) ---
  std::string logits2d = e.Gemm(tgt, "class_embed.weight", "class_embed.bias");  // [Q, C]
  std::string b0 =
      e.Unary("Relu", e.Gemm(tgt, "bbox_embed.layers.0.weight", "bbox_embed.layers.0.bias"));
  std::string b1 =
      e.Unary("Relu", e.Gemm(b0, "bbox_embed.layers.1.weight", "bbox_embed.layers.1.bias"));
  std::string b2 = e.Gemm(b1, "bbox_embed.layers.2.weight", "bbox_embed.layers.2.bias");
  std::string boxes2d = e.Unary("Sigmoid", e.Binary("Add", b2, "ref_pad"));

  std::string logits = e.Reshape(logits2d, {1, Q, C});
  std::string boxes = e.Reshape(boxes2d, {1, Q, 4});
  g.AddNode("Identity", {logits}, {"logits"});
  g.AddNode("Identity", {boxes}, {"boxes"});
  g.AddOutput("logits", {1, Q, C});
  g.AddOutput("boxes", {1, Q, 4});

  if (e.error) {
    return tl::make_unexpected(*e.error);
  }
  return g.Save(path, /*opset=*/17);
}

}  // namespace detr::onnxexport
