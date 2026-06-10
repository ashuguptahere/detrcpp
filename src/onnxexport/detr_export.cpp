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
#include <set>
#include <string>
#include <utility>
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
  std::set<std::string> added_weights;  // weights shared across layers add once

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
    if (!added_weights.insert(name).second) {
      return;  // a shared weight (e.g. a per-layer-applied MLP) — already emitted
    }
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

  // Multi-head attention, decomposed. Returns the [Lq, d] output. |d_over|/|nh_over|
  // override the model dim/heads (the ViT backbone runs at vit_embed/vit_heads).
  std::string Mha(const std::string& qin, const std::string& kin, const std::string& vin,
                  const std::string& prefix, int lq, int lk, int d_over = -1, int nh_over = -1) {
    const int d = d_over > 0 ? d_over : a.hidden_dim;
    const int nh = nh_over > 0 ? nh_over : a.nheads;
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

  std::string Slice(const std::string& x, std::int64_t start, std::int64_t end, std::int64_t axis,
                    std::int64_t step = 1) {
    const std::string st = g.Unique("st"), en = g.Unique("en"), ax = g.Unique("ax"),
                      sp = g.Unique("sp");
    AddShape(st, {start});
    AddShape(en, {end});
    AddShape(ax, {axis});
    AddShape(sp, {step});
    std::string y = g.Unique("slice");
    g.AddNode("Slice", {x, st, en, ax, sp}, {y});
    return y;
  }

  std::string Unsqueeze(const std::string& x, std::int64_t axis) {
    const std::string ax = g.Unique("uax");
    AddShape(ax, {axis});
    std::string y = g.Unique("unsqueeze");
    g.AddNode("Unsqueeze", {x, ax}, {y});
    return y;
  }

  // Multi-layer perceptron `prefix.layers.{0..n-1}` with relu between layers.
  std::string MlpN(const std::string& x, const std::string& prefix, int n) {
    std::string y = x;
    for (int i = 0; i < n; ++i) {
      y = Gemm(y, fmt::format("{}.layers.{}.weight", prefix, i),
               fmt::format("{}.layers.{}.bias", prefix, i));
      if (i + 1 < n) {
        y = Unary("Relu", y);
      }
    }
    return y;
  }

  std::string PRelu(const std::string& x, const std::string& slope) {
    AddWeight(slope);  // [1]
    std::string y = g.Unique("prelu");
    g.AddNode("PRelu", {x, slope}, {y});
    return y;
  }

  // InverseSigmoid: log(clamp(x,eps,1) / clamp(1-x,eps,1)), x first clamped [0,1].
  std::string InvSigmoid(const std::string& x) {
    const std::string c0 = g.Unique("c0"), c1 = g.Unique("c1"), eps = g.Unique("eps"),
                      one = g.Unique("one");
    AddFloatConst(c0, {1}, {0.0F});
    AddFloatConst(c1, {1}, {1.0F});
    AddFloatConst(eps, {1}, {1e-5F});
    AddFloatConst(one, {1}, {1.0F});
    std::string xc = Binary("Min", Binary("Max", x, c0), c1);  // clamp [0,1]
    std::string num = Binary("Max", xc, eps);
    std::string den = Binary("Max", Binary("Sub", one, xc), eps);
    return Unary("Log", Binary("Div", num, den));
  }

  // DAB's SineEmbed4D(obj[nq,4]) -> [nq,2d]: cat of embed(y),embed(x),embed(w),
  // embed(h) where embed interleaves sin(even)/cos(odd). |scaledim| = scale/dim_t
  // (temperature 10000) is a baked constant.
  std::string SineEmbed4DOps(const std::string& obj, int nq) {
    const int d = a.hidden_dim;
    const int half = d / 2;
    const double scale = 2.0 * std::numbers::pi;
    std::vector<float> scaledim(static_cast<std::size_t>(half));
    for (int k = 0; k < half; ++k) {
      scaledim[static_cast<std::size_t>(k)] = static_cast<float>(
          scale / std::pow(10000.0, 2.0 * std::floor(k / 2.0) / static_cast<double>(half)));
    }
    const std::string sd = g.Unique("scaledim");
    AddFloatConst(sd, {half}, scaledim);
    auto embed = [&](std::int64_t coord_idx) {
      std::string c = Slice(obj, coord_idx, coord_idx + 1, 1);  // [nq,1]
      std::string p = Binary("Mul", c, sd);                     // [nq,half]
      std::string s = Unsqueeze(Unary("Sin", Slice(p, 0, half, 1, 2)), 2);  // [nq,half/2,1]
      std::string co = Unsqueeze(Unary("Cos", Slice(p, 1, half, 1, 2)), 2);
      return Reshape(Concat({s, co}, 2), {nq, half});  // interleaved [nq,half]
    };
    return Concat({embed(1), embed(0), embed(2), embed(3)}, 1);  // [nq,2d]
  }

  // Bilinear grid sample. x:[N,C,H,W], grid:[N,Hout,Wout,2] in [-1,1] -> [N,C,Hout,Wout].
  std::string GridSample(const std::string& x, const std::string& grid) {
    std::string y = g.Unique("gridsample");
    g.AddNode("GridSample", {x, grid}, {y});
    g.AttrString("mode", "bilinear");  // opset 16/17 spelling
    g.AttrString("padding_mode", "zeros");
    g.AttrInt("align_corners", 0);
    return y;
  }

  std::string ReduceSum(const std::string& x, const std::vector<std::int64_t>& axes, bool keepdims) {
    std::string ax = g.Unique("rsax");
    AddShape(ax, axes);  // axes is an input initializer at opset 13+
    std::string y = g.Unique("reducesum");
    g.AddNode("ReduceSum", {x, ax}, {y});
    g.AttrInt("keepdims", keepdims ? 1 : 0);
    return y;
  }

  // GroupNorm via InstanceNormalization (opset-17 has no GroupNormalization):
  // normalize per group over (C/groups)*H*W, then the per-channel affine.
  std::string GroupNorm(const std::string& x, const std::string& prefix, int groups, int C, int H,
                        int W) {
    std::string xr = Reshape(x, {1, groups, (C / groups) * H * W});
    std::string sc = g.Unique("gn_s");
    AddFloatConst(sc, {groups}, std::vector<float>(static_cast<std::size_t>(groups), 1.0F));
    std::string bi = g.Unique("gn_b");
    AddFloatConst(bi, {groups}, std::vector<float>(static_cast<std::size_t>(groups), 0.0F));
    std::string norm = g.Unique("instnorm");
    g.AddNode("InstanceNormalization", {xr, sc, bi}, {norm});
    g.AttrFloat("epsilon", 1e-5F);
    std::string back = Reshape(norm, {1, C, H, W});
    AddWeight(prefix + ".weight");
    AddWeight(prefix + ".bias");
    std::string wr = Reshape(prefix + ".weight", {1, C, 1, 1});
    std::string br = Reshape(prefix + ".bias", {1, C, 1, 1});
    return Binary("Add", Binary("Mul", back, wr), br);
  }

  std::string ReduceMax(const std::string& x, std::int64_t axis, bool keepdims) {
    std::string y = g.Unique("reducemax");
    g.AddNode("ReduceMax", {x}, {y});
    g.AttrInts("axes", {axis});  // axes is an attribute at opset 17
    g.AttrInt("keepdims", keepdims ? 1 : 0);
    return y;
  }

  // TopK indices (largest, sorted) along |axis|. Returns the int64 indices output.
  std::string TopK(const std::string& x, std::int64_t k, std::int64_t axis) {
    std::string kc = g.Unique("topk_k");
    g.AddInitializerI64(kc, {1}, {k});
    std::string vals = g.Unique("topk_v"), idx = g.Unique("topk_i");
    g.AddNode("TopK", {x, kc}, {vals, idx});
    g.AttrInt("axis", axis);
    g.AttrInt("largest", 1);
    g.AttrInt("sorted", 1);
    return idx;
  }

  std::string Gather(const std::string& data, const std::string& indices, std::int64_t axis) {
    std::string y = g.Unique("gather");
    g.AddNode("Gather", {data, indices}, {y});
    g.AttrInt("axis", axis);
    return y;
  }

  std::string Expand(const std::string& x, const std::vector<std::int64_t>& dims) {
    std::string sh = g.Unique("exshape");
    AddShape(sh, dims);
    std::string y = g.Unique("expand");
    g.AddNode("Expand", {x, sh}, {y});
    return y;
  }

  // --- RT-DETR helpers (ResNet-VD backbone, CCFM, AIFI) ---

  std::string AvgPool(const std::string& x, int k, int stride) {
    std::string y = g.Unique("avgpool");
    g.AddNode("AveragePool", {x}, {y});
    g.AttrInts("kernel_shape", {k, k});
    g.AttrInts("strides", {stride, stride});
    g.AttrInt("ceil_mode", 1);
    return y;
  }

  std::string Resize2xNearest(const std::string& x) {
    std::string scales = g.Unique("scales");
    AddFloatConst(scales, {4}, {1.0F, 1.0F, 2.0F, 2.0F});
    std::string y = g.Unique("resize");
    g.AddNode("Resize", {x, "", scales}, {y});  // "" = empty roi
    g.AttrString("mode", "nearest");
    g.AttrString("coordinate_transformation_mode", "asymmetric");
    g.AttrString("nearest_mode", "floor");
    return y;
  }

  std::string Silu(const std::string& x) { return Binary("Mul", x, Unary("Sigmoid", x)); }

  // Exact (erf) GELU: 0.5*x*(1+erf(x/sqrt2)).
  std::string Gelu(const std::string& x) {
    std::string inv = g.Unique("gelu_inv");
    AddFloatConst(inv, {1}, {static_cast<float>(1.0 / std::sqrt(2.0))});
    std::string half = g.Unique("gelu_half");
    AddFloatConst(half, {1}, {0.5F});
    std::string one = g.Unique("gelu_one");
    AddFloatConst(one, {1}, {1.0F});
    std::string er = Unary("Erf", Binary("Mul", x, inv));
    return Binary("Mul", Binary("Mul", x, half), Binary("Add", one, er));
  }

  // Conv(bias=false) + BatchNorm + optional SiLU (RT-DETR ConvNormLayer .conv/.norm).
  std::string ConvNorm(const std::string& x, const std::string& p, int k, int stride, bool act) {
    std::string y = Bn(Conv(x, p + ".conv.weight", "", k, stride, (k - 1) / 2), p + ".norm");
    return act ? Silu(y) : y;
  }

  std::string RepVgg(const std::string& x, const std::string& p) {
    return Silu(Binary("Add", ConvNorm(x, p + ".conv1", 3, 1, false), ConvNorm(x, p + ".conv2", 1, 1, false)));
  }

  std::string CSPRep(const std::string& x, const std::string& p, int num_blocks) {
    std::string x1 = ConvNorm(x, p + ".conv1", 1, 1, true);
    for (int i = 0; i < num_blocks; ++i) {
      x1 = RepVgg(x1, fmt::format("{}.bottlenecks.{}", p, i));
    }
    return Binary("Add", x1, ConvNorm(x, p + ".conv2", 1, 1, true));
  }

  // ResNet-D/VD bottleneck: deep stem upstream; stride-2 blocks use an AvgPool +
  // 1x1 conv shortcut (downsample.{1,2}); the stage-0 channel-only block a plain
  // 1x1 conv (downsample.{0,1}).
  std::string BottleneckVD(const std::string& x, const std::string& p, int stride, bool has_down) {
    std::string c = Unary("Relu", Bn(Conv(x, p + ".conv1.weight", "", 1, 1, 0), p + ".bn1"));
    c = Unary("Relu", Bn(Conv(c, p + ".conv2.weight", "", 3, stride, 1), p + ".bn2"));
    c = Bn(Conv(c, p + ".conv3.weight", "", 1, 1, 0), p + ".bn3");
    std::string id = x;
    if (has_down) {
      if (stride > 1) {
        id = Bn(Conv(AvgPool(x, 2, 2), p + ".downsample.1.weight", "", 1, 1, 0), p + ".downsample.2");
      } else {
        id = Bn(Conv(x, p + ".downsample.0.weight", "", 1, 1, 0), p + ".downsample.1");
      }
    }
    return Unary("Relu", Binary("Add", c, id));
  }

  std::array<std::string, 3> ResNetVDStages(const std::string& input) {
    const int stem_stride[3] = {2, 1, 1};
    std::string x = input;
    for (int s = 0; s < 3; ++s) {
      x = Unary("Relu", Bn(Conv(x, fmt::format("backbone.stem.{}.0.weight", s), "", 3, stem_stride[s], 1),
                           fmt::format("backbone.stem.{}.1", s)));
    }
    x = Pool(x, 3, 2, 1);
    const auto& blocks = a.resnet_blocks;
    const int strides[4] = {1, 2, 2, 2};
    std::array<std::string, 3> outs;
    for (int lyr = 0; lyr < 4; ++lyr) {
      for (int b = 0; b < blocks[static_cast<std::size_t>(lyr)]; ++b) {
        const std::string p = fmt::format("backbone.layer{}.{}", lyr + 1, b);
        const int stride = (b == 0) ? strides[lyr] : 1;
        x = BottleneckVD(x, p, stride, /*has_down=*/b == 0);
      }
      if (lyr >= 1) {
        outs[static_cast<std::size_t>(lyr - 1)] = x;
      }
    }
    return outs;
  }

  // RT-DETR AIFI 2D sin-cos position [HW, dim] (temperature 10000), replicating
  // SinCos2d's meshgrid("ij") flatten exactly (w-major: k = iw*h + jh).
  std::vector<float> SinCos2d(int h, int w, int dim) {
    const int pd = dim / 4;
    std::vector<double> omega(static_cast<std::size_t>(pd));
    for (int k = 0; k < pd; ++k) {
      omega[static_cast<std::size_t>(k)] = 1.0 / std::pow(10000.0, static_cast<double>(k) / pd);
    }
    std::vector<float> pos(static_cast<std::size_t>(h) * static_cast<std::size_t>(w) *
                           static_cast<std::size_t>(dim));
    for (int iw = 0; iw < w; ++iw) {
      for (int jh = 0; jh < h; ++jh) {
        const auto base = static_cast<std::size_t>(iw * h + jh) * static_cast<std::size_t>(dim);
        for (int c = 0; c < pd; ++c) {
          pos[base + static_cast<std::size_t>(c)] = static_cast<float>(std::sin(iw * omega[static_cast<std::size_t>(c)]));
          pos[base + static_cast<std::size_t>(pd + c)] = static_cast<float>(std::cos(iw * omega[static_cast<std::size_t>(c)]));
          pos[base + static_cast<std::size_t>(2 * pd + c)] = static_cast<float>(std::sin(jh * omega[static_cast<std::size_t>(c)]));
          pos[base + static_cast<std::size_t>(3 * pd + c)] = static_cast<float>(std::cos(jh * omega[static_cast<std::size_t>(c)]));
        }
      }
    }
    return pos;
  }

  // ViT 2D sin-cos position [HW, dim]: meshgrid("ij") on (h,w) -> H-major flatten,
  // cat(sin(h),cos(h),sin(w),cos(w)). Matches the tokens' flatten order (no shift).
  std::vector<float> SinCos2dVit(int h, int w, int dim) {
    const int pd = dim / 4;
    std::vector<double> omega(static_cast<std::size_t>(pd));
    for (int k = 0; k < pd; ++k) {
      omega[static_cast<std::size_t>(k)] = 1.0 / std::pow(10000.0, static_cast<double>(k) / pd);
    }
    std::vector<float> pos(static_cast<std::size_t>(h) * static_cast<std::size_t>(w) *
                           static_cast<std::size_t>(dim));
    for (int ih = 0; ih < h; ++ih) {
      for (int jw = 0; jw < w; ++jw) {
        const auto base = static_cast<std::size_t>(ih * w + jw) * static_cast<std::size_t>(dim);
        for (int c = 0; c < pd; ++c) {
          pos[base + static_cast<std::size_t>(c)] = static_cast<float>(std::sin(ih * omega[static_cast<std::size_t>(c)]));
          pos[base + static_cast<std::size_t>(pd + c)] = static_cast<float>(std::cos(ih * omega[static_cast<std::size_t>(c)]));
          pos[base + static_cast<std::size_t>(2 * pd + c)] = static_cast<float>(std::sin(jw * omega[static_cast<std::size_t>(c)]));
          pos[base + static_cast<std::size_t>(3 * pd + c)] = static_cast<float>(std::cos(jw * omega[static_cast<std::size_t>(c)]));
        }
      }
    }
    return pos;
  }

  // ViT backbone: patch embed conv + pre-norm transformer blocks (GELU FFN) ->
  // [1, vit_embed, h, w] (h=w=imgsz/patch).
  std::string ViTBackbone(const std::string& input, int h, int w) {
    const int dim = a.vit_embed;
    const int nh = a.vit_heads;
    const int L = h * w;
    std::string x = Conv(input, "backbone.patch_embed.weight", "backbone.patch_embed.bias", a.patch,
                         a.patch, 0);             // [1, dim, h, w]
    std::string tok = Transpose(Reshape(x, {dim, L}), {1, 0});  // [L, dim]
    AddFloatConst("vit_pos", {L, dim}, SinCos2dVit(h, w, dim));
    for (int i = 0; i < a.vit_depth; ++i) {
      const std::string p = fmt::format("backbone.blocks.{}", i);
      std::string n = LayerNorm(tok, p + ".norm1");
      std::string att = Mha(Binary("Add", n, "vit_pos"), Binary("Add", n, "vit_pos"), n, p + ".attn",
                            L, L, dim, nh);
      tok = Binary("Add", tok, att);
      std::string ff = Gemm(Gelu(Gemm(LayerNorm(tok, p + ".norm2"), p + ".fc1.weight", p + ".fc1.bias")),
                            p + ".fc2.weight", p + ".fc2.bias");
      tok = Binary("Add", tok, ff);
    }
    tok = LayerNorm(tok, "backbone.norm");
    return Reshape(Transpose(tok, {1, 0}), {1, dim, h, w});
  }

  // Multi-scale deformable attention (MSDeformAttnImpl::forward + Core), batch 1.
  // query [Lq,d]; reference [Lq,nl,2] (2D) or [Lq,nl,4] (4D); value_src [Lv,d].
  std::string MSDeformAttn(const std::string& query, const std::string& reference,
                           const std::string& value_src,
                           const std::vector<std::pair<int, int>>& shapes, const std::string& p,
                           int Lq, bool ref4d) {
    const int d = a.hidden_dim;
    const int nh = a.nheads;
    const int hd = d / nh;
    const int nl = static_cast<int>(shapes.size());
    const int np = a.num_points;
    int Lv = 0;
    for (const auto& [h, w] : shapes) {
      Lv += h * w;
    }
    // (A) value_proj -> per-level [nh,hd,H,W] maps.
    std::string value_mh =
        Reshape(Gemm(value_src, p + ".value_proj.weight", p + ".value_proj.bias"), {Lv, nh, hd});
    std::vector<std::string> v_levels;
    int start = 0;
    for (int l = 0; l < nl; ++l) {
      const int h = shapes[static_cast<std::size_t>(l)].first;
      const int w = shapes[static_cast<std::size_t>(l)].second;
      std::string v_l = Transpose(Slice(value_mh, start, start + h * w, 0), {1, 2, 0});  // [nh,hd,hw]
      v_levels.push_back(Reshape(v_l, {nh, hd, h, w}));
      start += h * w;
    }
    // (B) sampling offsets + attention weights.
    std::string off = Reshape(
        Gemm(query, p + ".sampling_offsets.weight", p + ".sampling_offsets.bias"), {Lq, nh, nl, np, 2});
    std::string aw = Reshape(
        Gemm(query, p + ".attention_weights.weight", p + ".attention_weights.bias"), {Lq, nh, nl * np});
    aw = Reshape(Softmax(aw, -1), {Lq, nh, nl, np});
    // (C) sampling locations -> grid in [-1,1].
    std::string loc;
    if (!ref4d) {
      std::string ref6 = Reshape(reference, {Lq, 1, nl, 1, 2});
      std::vector<float> onorm;  // (W,H) per level
      for (int l = 0; l < nl; ++l) {
        onorm.push_back(static_cast<float>(shapes[static_cast<std::size_t>(l)].second));
        onorm.push_back(static_cast<float>(shapes[static_cast<std::size_t>(l)].first));
      }
      std::string onorm5 = g.Unique("onorm");
      AddFloatConst(onorm5, {1, 1, nl, 1, 2}, onorm);
      loc = Binary("Add", ref6, Binary("Div", off, onorm5));
    } else {
      std::string ref6 = Reshape(reference, {Lq, 1, nl, 1, 4});
      std::string invnp = g.Unique("invnp");
      AddFloatConst(invnp, {1}, {1.0F / static_cast<float>(np)});
      std::string half = g.Unique("half");
      AddFloatConst(half, {1}, {0.5F});
      loc = Binary("Add", Slice(ref6, 0, 2, -1),
                   Binary("Mul", Binary("Mul", Binary("Mul", off, invnp), Slice(ref6, 2, 4, -1)), half));
    }
    std::string two = g.Unique("two");
    AddFloatConst(two, {1}, {2.0F});
    std::string one = g.Unique("one");
    AddFloatConst(one, {1}, {1.0F});
    std::string grid_all = Binary("Sub", Binary("Mul", loc, two), one);  // [Lq,nh,nl,np,2]
    // (D) per-level grid sample.
    std::vector<std::string> samp;
    for (int l = 0; l < nl; ++l) {
      std::string g_l = Reshape(Slice(grid_all, l, l + 1, 2), {Lq, nh, np, 2});
      samp.push_back(GridSample(v_levels[static_cast<std::size_t>(l)],
                                Transpose(g_l, {1, 0, 2, 3})));  // [nh,hd,Lq,np]
    }
    // (E) weighted sum over levels*points.
    std::vector<std::string> su;
    su.reserve(samp.size());
    for (const auto& s : samp) {
      su.push_back(Unsqueeze(s, 3));  // [nh,hd,Lq,1,np]
    }
    std::string stacked = Reshape(Concat(su, 3), {nh, hd, Lq, nl * np});
    std::string attn_r = Reshape(Transpose(aw, {1, 0, 2, 3}), {nh, 1, Lq, nl * np});
    std::string summed = ReduceSum(Binary("Mul", stacked, attn_r), {3}, /*keepdims=*/false);
    std::string out = Reshape(Transpose(Reshape(summed, {1, nh * hd, Lq}), {0, 2, 1}), {Lq, d});
    return Gemm(out, p + ".output_proj.weight", p + ".output_proj.bias");
  }

  // Deformable-DETR baked constants: encoder pos+level_embed, encoder reference
  // grid, the fixed query reference, and the inverse-sigmoid box padding.
  struct DeformConsts {
    std::vector<float> pos_cat;    // [Lv, d]
    std::vector<float> enc_ref;    // [Lv, nl, 2]
    std::vector<float> dec_ref;    // [nq, nl, 2]
    std::vector<float> ref_pad;    // [nq, 4]
  };
  DeformConsts DeformPrecompute(const std::vector<std::pair<int, int>>& shapes) {
    const int d = a.hidden_dim;
    const int nl = static_cast<int>(shapes.size());
    const int nq = a.num_queries;
    DeformConsts c;
    int Lv = 0;
    for (const auto& [h, w] : shapes) {
      Lv += h * w;
    }
    const RawTensor* le = Need("level_embed");
    if (le == nullptr) {
      return c;
    }
    const auto LE = Floats(*le);
    c.enc_ref.assign(static_cast<std::size_t>(Lv) * nl * 2, 0.0F);
    c.pos_cat.assign(static_cast<std::size_t>(Lv) * d, 0.0F);
    int base = 0;
    for (int l = 0; l < nl; ++l) {
      const int h = shapes[static_cast<std::size_t>(l)].first;
      const int w = shapes[static_cast<std::size_t>(l)].second;
      const auto sine = SinePos(h, w);  // [h*w*d], shared DETR sine (matches deformable)
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          const int lv = base + i * w + j;
          const float xx = (static_cast<float>(j) + 0.5F) / static_cast<float>(w);
          const float yy = (static_cast<float>(i) + 0.5F) / static_cast<float>(h);
          for (int ll = 0; ll < nl; ++ll) {
            c.enc_ref[static_cast<std::size_t>((lv * nl + ll) * 2 + 0)] = xx;
            c.enc_ref[static_cast<std::size_t>((lv * nl + ll) * 2 + 1)] = yy;
          }
          for (int cc = 0; cc < d; ++cc) {
            c.pos_cat[static_cast<std::size_t>(lv * d + cc)] =
                sine[static_cast<std::size_t>((i * w + j) * d + cc)] +
                LE[static_cast<std::size_t>(l * d + cc)];
          }
        }
      }
      base += h * w;
    }
    // Optional: deformable-detr's single-stage query reference (absent in dino,
    // which selects queries via topk in the deform head) — Find, don't Need.
    const RawTensor* qe = sd.Find("query_embed.weight");       // [nq, 2d]
    const RawTensor* rw = sd.Find("reference_points.weight");  // [2, d]
    const RawTensor* rb = sd.Find("reference_points.bias");    // [2]
    if (qe == nullptr || rw == nullptr || rb == nullptr) {
      return c;
    }
    const auto QE = Floats(*qe), RW = Floats(*rw), RB = Floats(*rb);
    c.dec_ref.assign(static_cast<std::size_t>(nq) * nl * 2, 0.0F);
    c.ref_pad.assign(static_cast<std::size_t>(nq) * 4, 0.0F);
    auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    auto inv = [](double s) {
      const double e = 1e-5;
      return std::log(std::max(s, e) / std::max(1.0 - s, e));
    };
    for (int q = 0; q < nq; ++q) {
      double r[2];
      for (int o = 0; o < 2; ++o) {
        double s = RB[static_cast<std::size_t>(o)];
        for (int i = 0; i < d; ++i) {
          s += static_cast<double>(RW[static_cast<std::size_t>(o * d + i)]) *
               static_cast<double>(QE[static_cast<std::size_t>(q * 2 * d + i)]);
        }
        r[o] = sig(s);
      }
      for (int ll = 0; ll < nl; ++ll) {
        c.dec_ref[static_cast<std::size_t>((q * nl + ll) * 2 + 0)] = static_cast<float>(r[0]);
        c.dec_ref[static_cast<std::size_t>((q * nl + ll) * 2 + 1)] = static_cast<float>(r[1]);
      }
      c.ref_pad[static_cast<std::size_t>(q * 4 + 0)] = static_cast<float>(inv(r[0]));
      c.ref_pad[static_cast<std::size_t>(q * 4 + 1)] = static_cast<float>(inv(r[1]));
    }
    return c;
  }

  // GenerateAnchors: grid-center anchors in inverse-sigmoid space, [Lv,4], with
  // invalid (near-border) locations set to 1e9. A constant given |shapes|.
  std::vector<float> DeformAnchors(const std::vector<std::pair<int, int>>& shapes) {
    int Lv = 0;
    for (const auto& [h, w] : shapes) {
      Lv += h * w;
    }
    std::vector<float> a4(static_cast<std::size_t>(Lv) * 4);
    int base = 0;
    for (int l = 0; l < static_cast<int>(shapes.size()); ++l) {
      const int h = shapes[static_cast<std::size_t>(l)].first;
      const int w = shapes[static_cast<std::size_t>(l)].second;
      const double wh = 0.05 * std::pow(2.0, l);
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          const int lv = base + i * w + j;
          const double comp[4] = {(j + 0.5) / w, (i + 0.5) / h, wh, wh};
          bool valid = true;
          for (double cv : comp) {
            if (!(cv > 1e-2 && cv < 1.0 - 1e-2)) {
              valid = false;
            }
          }
          for (int k = 0; k < 4; ++k) {
            a4[static_cast<std::size_t>(lv * 4 + k)] =
                valid ? static_cast<float>(std::log(comp[k] / (1.0 - comp[k]))) : 1e9F;
          }
        }
      }
      base += h * w;
    }
    return a4;
  }

  // Shared deformable detection head (RunDeformDetectHead): query selection via
  // topk over the encoder classification, then an iterative-refinement deformable
  // decoder (4D refs). Returns {logits[nq,C], boxes[nq,4]} from the last layer.
  std::pair<std::string, std::string> DeformHead(const std::string& memory,
                                                 const std::vector<std::pair<int, int>>& shapes) {
    const int d = a.hidden_dim;
    const int C = a.num_classes;
    const int nq = a.num_queries;
    const int nl = static_cast<int>(shapes.size());
    int Lv = 0;
    for (const auto& [h, w] : shapes) {
      Lv += h * w;
    }
    AddFloatConst("anchors", {Lv, 4}, DeformAnchors(shapes));
    std::string out_mem =
        LayerNorm(Gemm(memory, "enc_output.weight", "enc_output.bias"), "enc_output_norm");
    std::string enc_class = Gemm(out_mem, "enc_score_head.weight", "enc_score_head.bias");  // [Lv,C]
    std::string enc_coord = Binary("Add", MlpN(out_mem, "enc_bbox_head", 3), "anchors");    // [Lv,4]
    // Query selection: topk over the max-over-classes score.
    std::string cls_max = Reshape(ReduceMax(enc_class, 1, /*keepdims=*/false), {1, Lv});
    std::string idx = Reshape(TopK(cls_max, nq, 1), {nq});
    std::string ref = Unary("Sigmoid", Gather(enc_coord, idx, 0));  // [nq,4]
    std::string tgt = Gather(out_mem, idx, 0);                      // [nq,d]
    std::string logits;
    std::string boxes;
    for (int i = 0; i < a.dec_layers; ++i) {
      const std::string p = fmt::format("decoder.{}", i);
      std::string ref_input = Expand(Unsqueeze(ref, 1), {nq, nl, 4});  // [nq,nl,4]
      std::string query_pos = MlpN(ref, "query_pos_head", 2);          // [nq,d]
      std::string q = Binary("Add", tgt, query_pos);
      std::string sa = Mha(q, q, tgt, p + ".self_attn", nq, nq);
      tgt = LayerNorm(Binary("Add", tgt, sa), p + ".norm1");
      std::string ca = MSDeformAttn(Binary("Add", tgt, query_pos), ref_input, memory, shapes,
                                    p + ".cross_attn", nq, /*ref4d=*/true);
      tgt = LayerNorm(Binary("Add", tgt, ca), p + ".norm2");
      std::string ff = Gemm(Unary("Relu", Gemm(tgt, p + ".linear1.weight", p + ".linear1.bias")),
                            p + ".linear2.weight", p + ".linear2.bias");
      tgt = LayerNorm(Binary("Add", tgt, ff), p + ".norm3");
      boxes = Unary("Sigmoid", Binary("Add", MlpN(tgt, fmt::format("dec_bbox_head.{}", i), 3),
                                      InvSigmoid(ref)));
      logits = Gemm(tgt, fmt::format("dec_score_head.{}.weight", i),
                    fmt::format("dec_score_head.{}.bias", i));
      ref = boxes;
    }
    return {logits, boxes};
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
                        const std::string& query_sine, bool is_first, int Q, int L,
                        bool prelu = false) {
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
    // FFN (conditional: ReLU; DAB: learnable PReLU).
    std::string h1 = Gemm(tgt, p + ".linear1.weight", p + ".linear1.bias");
    std::string act = prelu ? PRelu(h1, p + ".activation_fn.weight") : Unary("Relu", h1);
    std::string ff = Gemm(act, p + ".linear2.weight", p + ".linear2.bias");
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

  // {C3, C4, C5} = layer2/layer3/layer4 outputs (the deformable multi-tap).
  std::array<std::string, 3> ResNet50Stages(const std::string& input) {
    std::string x =
        Unary("Relu", Bn(Conv(input, "backbone.conv1.weight", "", 7, 2, 3), "backbone.bn1"));
    x = Pool(x, 3, 2, 1);
    const auto& blocks = a.resnet_blocks;
    const int strides[4] = {1, 2, 2, 2};
    std::array<std::string, 3> outs;
    for (int lyr = 0; lyr < 4; ++lyr) {
      for (int b = 0; b < blocks[static_cast<std::size_t>(lyr)]; ++b) {
        const std::string p = fmt::format("backbone.layer{}.{}", lyr + 1, b);
        const int stride = (b == 0) ? strides[lyr] : 1;
        x = Bottleneck(x, p, stride, /*has_down=*/b == 0);
      }
      if (lyr >= 1) {
        outs[static_cast<std::size_t>(lyr - 1)] = x;  // layer2->C3, layer3->C4, layer4->C5
      }
    }
    return outs;
  }

  std::string ResNet50Backbone(const std::string& input) { return ResNet50Stages(input)[2]; }

  // Replicates SinePos -> a constant [L, d] positional encoding (L = h*w).
  // |temp| is the frequency base (DETR/conditional 10000; DAB 20).
  std::vector<float> SinePos(int h, int w, double temp = 10000.0) {
    const int d = a.hidden_dim;
    const int half = d / 2;
    const double scale = 2.0 * std::numbers::pi;
    std::vector<double> dim_t(static_cast<std::size_t>(half));
    for (int k = 0; k < half; ++k) {
      dim_t[static_cast<std::size_t>(k)] =
          std::pow(temp, 2.0 * std::floor(k / 2.0) / static_cast<double>(half));
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

core::Result<void> ExportDeformable(const DetrArch& arch, const weights::StateDict& weights,
                                    const std::string& path) {
  GraphBuilder g("deformable-detr");
  Emitter e{g, weights, arch, std::nullopt};

  const int d = arch.hidden_dim;
  const int C = arch.num_classes;
  const int nq = arch.num_queries;
  const int nl = arch.num_levels;
  g.AddInput("images", {1, 3, arch.imgsz, arch.imgsz});

  // --- backbone + 4-level input_proj (Conv + GroupNorm) ---
  auto stages = e.ResNet50Stages("images");  // {C3, C4, C5}
  std::vector<std::string> proj;
  std::vector<std::pair<int, int>> shapes;
  const int fs[3] = {arch.imgsz / 8, arch.imgsz / 16, arch.imgsz / 32};
  for (int i = 0; i < 3; ++i) {
    std::string c = e.Conv(stages[static_cast<std::size_t>(i)], fmt::format("input_proj.{}.0.weight", i),
                           fmt::format("input_proj.{}.0.bias", i), 1, 1, 0);
    proj.push_back(e.GroupNorm(c, fmt::format("input_proj.{}.1", i), 32, d, fs[i], fs[i]));
    shapes.emplace_back(fs[i], fs[i]);
  }
  int prev_sp = fs[2];
  for (int i = 3; i < nl; ++i) {
    std::string in = (i == 3) ? stages[2] : proj.back();
    std::string c = e.Conv(in, fmt::format("input_proj.{}.0.weight", i),
                           fmt::format("input_proj.{}.0.bias", i), 3, 2, 1);
    const int sp = (prev_sp - 1) / 2 + 1;
    proj.push_back(e.GroupNorm(c, fmt::format("input_proj.{}.1", i), 32, d, sp, sp));
    shapes.emplace_back(sp, sp);
    prev_sp = sp;
  }

  // --- flatten levels into one memory; pos (sine + level_embed) and the encoder
  // reference grid and the fixed query reference are all baked constants ---
  const Emitter::DeformConsts cc = e.DeformPrecompute(shapes);
  int Lv = 0;
  std::vector<std::string> src_flat;
  for (int l = 0; l < nl; ++l) {
    const int h = shapes[static_cast<std::size_t>(l)].first;
    const int w = shapes[static_cast<std::size_t>(l)].second;
    src_flat.push_back(e.Transpose(e.Reshape(proj[static_cast<std::size_t>(l)], {d, h * w}), {1, 0}));
    Lv += h * w;
  }
  std::string memory = e.Concat(src_flat, 0);  // [Lv, d]
  e.AddFloatConst("pos_cat", {Lv, d}, cc.pos_cat);
  e.AddFloatConst("enc_ref", {Lv, nl, 2}, cc.enc_ref);
  e.AddFloatConst("dec_ref", {nq, nl, 2}, cc.dec_ref);
  e.AddFloatConst("ref_pad", {nq, 4}, cc.ref_pad);

  // --- deformable encoder ---
  for (int i = 0; i < arch.enc_layers; ++i) {
    const std::string p = fmt::format("encoder.{}", i);
    std::string src2 = e.MSDeformAttn(e.Binary("Add", memory, "pos_cat"), "enc_ref", memory, shapes,
                                      p + ".self_attn", Lv, /*ref4d=*/false);
    memory = e.LayerNorm(e.Binary("Add", memory, src2), p + ".norm1");
    std::string ff = e.Gemm(e.Unary("Relu", e.Gemm(memory, p + ".linear1.weight", p + ".linear1.bias")),
                            p + ".linear2.weight", p + ".linear2.bias");
    memory = e.LayerNorm(e.Binary("Add", memory, ff), p + ".norm2");
  }

  // --- query init + deformable decoder ---
  e.AddWeight("query_embed.weight");                                // [nq, 2d]
  std::string query_pos = e.Slice("query_embed.weight", 0, d, 1);   // [nq, d]
  std::string tgt = e.Slice("query_embed.weight", d, 2 * d, 1);     // [nq, d]
  for (int i = 0; i < arch.dec_layers; ++i) {
    const std::string p = fmt::format("decoder.{}", i);
    std::string q = e.Binary("Add", tgt, query_pos);
    std::string sa = e.Mha(q, q, tgt, p + ".self_attn", nq, nq);
    tgt = e.LayerNorm(e.Binary("Add", tgt, sa), p + ".norm1");
    std::string ca = e.MSDeformAttn(e.Binary("Add", tgt, query_pos), "dec_ref", memory, shapes,
                                    p + ".cross_attn", nq, /*ref4d=*/false);
    tgt = e.LayerNorm(e.Binary("Add", tgt, ca), p + ".norm2");
    std::string ff = e.Gemm(e.Unary("Relu", e.Gemm(tgt, p + ".linear1.weight", p + ".linear1.bias")),
                            p + ".linear2.weight", p + ".linear2.bias");
    tgt = e.LayerNorm(e.Binary("Add", tgt, ff), p + ".norm3");
  }

  // --- heads (focal; box = sigmoid(mlp + ref_pad)) ---
  std::string logits2d = e.Gemm(tgt, "class_embed.weight", "class_embed.bias");  // [nq, C]
  std::string b0 = e.Unary("Relu", e.Gemm(tgt, "bbox_embed.0.weight", "bbox_embed.0.bias"));
  std::string b1 = e.Unary("Relu", e.Gemm(b0, "bbox_embed.2.weight", "bbox_embed.2.bias"));
  std::string b2 = e.Gemm(b1, "bbox_embed.4.weight", "bbox_embed.4.bias");
  std::string boxes2d = e.Unary("Sigmoid", e.Binary("Add", b2, "ref_pad"));

  std::string logits = e.Reshape(logits2d, {1, nq, C});
  std::string boxes = e.Reshape(boxes2d, {1, nq, 4});
  g.AddNode("Identity", {logits}, {"logits"});
  g.AddNode("Identity", {boxes}, {"boxes"});
  g.AddOutput("logits", {1, nq, C});
  g.AddOutput("boxes", {1, nq, 4});

  if (e.error) {
    return tl::make_unexpected(*e.error);
  }
  return g.Save(path, /*opset=*/17);
}

core::Result<void> ExportDino(const DetrArch& arch, const weights::StateDict& weights,
                              const std::string& path) {
  GraphBuilder g("dino");
  Emitter e{g, weights, arch, std::nullopt};

  const int d = arch.hidden_dim;
  const int C = arch.num_classes;
  const int nq = arch.num_queries;
  const int nl = arch.num_levels;
  g.AddInput("images", {1, 3, arch.imgsz, arch.imgsz});

  // --- backbone + 4-level input_proj (Conv + GroupNorm), same as deformable ---
  auto stages = e.ResNet50Stages("images");
  std::vector<std::string> proj;
  std::vector<std::pair<int, int>> shapes;
  const int fs[3] = {arch.imgsz / 8, arch.imgsz / 16, arch.imgsz / 32};
  for (int i = 0; i < 3; ++i) {
    std::string c = e.Conv(stages[static_cast<std::size_t>(i)], fmt::format("input_proj.{}.0.weight", i),
                           fmt::format("input_proj.{}.0.bias", i), 1, 1, 0);
    proj.push_back(e.GroupNorm(c, fmt::format("input_proj.{}.1", i), 32, d, fs[i], fs[i]));
    shapes.emplace_back(fs[i], fs[i]);
  }
  int prev_sp = fs[2];
  for (int i = 3; i < nl; ++i) {
    std::string in = (i == 3) ? stages[2] : proj.back();
    std::string c = e.Conv(in, fmt::format("input_proj.{}.0.weight", i),
                           fmt::format("input_proj.{}.0.bias", i), 3, 2, 1);
    const int sp = (prev_sp - 1) / 2 + 1;
    proj.push_back(e.GroupNorm(c, fmt::format("input_proj.{}.1", i), 32, d, sp, sp));
    shapes.emplace_back(sp, sp);
    prev_sp = sp;
  }

  // --- flatten + baked pos/enc_ref + deformable encoder ---
  const Emitter::DeformConsts cc = e.DeformPrecompute(shapes);
  int Lv = 0;
  std::vector<std::string> src_flat;
  for (int l = 0; l < nl; ++l) {
    const int h = shapes[static_cast<std::size_t>(l)].first;
    const int w = shapes[static_cast<std::size_t>(l)].second;
    src_flat.push_back(e.Transpose(e.Reshape(proj[static_cast<std::size_t>(l)], {d, h * w}), {1, 0}));
    Lv += h * w;
  }
  std::string memory = e.Concat(src_flat, 0);
  e.AddFloatConst("pos_cat", {Lv, d}, cc.pos_cat);
  e.AddFloatConst("enc_ref", {Lv, nl, 2}, cc.enc_ref);
  for (int i = 0; i < arch.enc_layers; ++i) {
    const std::string p = fmt::format("encoder.{}", i);
    std::string src2 = e.MSDeformAttn(e.Binary("Add", memory, "pos_cat"), "enc_ref", memory, shapes,
                                      p + ".self_attn", Lv, /*ref4d=*/false);
    memory = e.LayerNorm(e.Binary("Add", memory, src2), p + ".norm1");
    std::string ff = e.Gemm(e.Unary("Relu", e.Gemm(memory, p + ".linear1.weight", p + ".linear1.bias")),
                            p + ".linear2.weight", p + ".linear2.bias");
    memory = e.LayerNorm(e.Binary("Add", memory, ff), p + ".norm2");
  }

  // --- deformable detection head (topk query selection + decoder) ---
  auto heads = e.DeformHead(memory, shapes);
  std::string logits = e.Reshape(heads.first, {1, nq, C});
  std::string boxes = e.Reshape(heads.second, {1, nq, 4});
  g.AddNode("Identity", {logits}, {"logits"});
  g.AddNode("Identity", {boxes}, {"boxes"});
  g.AddOutput("logits", {1, nq, C});
  g.AddOutput("boxes", {1, nq, 4});

  if (e.error) {
    return tl::make_unexpected(*e.error);
  }
  return g.Save(path, /*opset=*/17);
}

core::Result<void> ExportRtDetr(const DetrArch& arch, const weights::StateDict& weights,
                                const std::string& path) {
  GraphBuilder g("rt-detr");
  Emitter e{g, weights, arch, std::nullopt};

  const int d = arch.hidden_dim;
  const int C = arch.num_classes;
  const int nq = arch.num_queries;
  const int nl = arch.num_levels;  // 3 backbone levels
  g.AddInput("images", {1, 3, arch.imgsz, arch.imgsz});

  // --- ResNet-VD backbone + input_proj (1x1 conv + BN, no GroupNorm) ---
  auto stages = e.ResNetVDStages("images");  // {C3, C4, C5}
  std::vector<std::string> proj;
  std::vector<std::pair<int, int>> shapes;
  const int fs[3] = {arch.imgsz / 8, arch.imgsz / 16, arch.imgsz / 32};
  for (int i = 0; i < nl; ++i) {
    std::string c = e.Conv(stages[static_cast<std::size_t>(i)], fmt::format("input_proj.{}.0.weight", i),
                           "", 1, 1, 0);
    proj.push_back(e.Bn(c, fmt::format("input_proj.{}.1", i)));
    shapes.emplace_back(fs[i], fs[i]);
  }

  // --- AIFI on the top level (GELU FFN, 2D sin-cos pos) ---
  {
    const int ht = fs[nl - 1];
    std::string src = e.Transpose(e.Reshape(proj[static_cast<std::size_t>(nl - 1)], {d, ht * ht}), {1, 0});
    e.AddFloatConst("aifi_pos", {ht * ht, d}, e.SinCos2d(ht, ht, d));
    for (int i = 0; i < arch.enc_layers; ++i) {
      const std::string p = fmt::format("aifi.{}", i);
      std::string q = e.Binary("Add", src, "aifi_pos");
      std::string attn = e.Mha(q, q, src, p + ".self_attn", ht * ht, ht * ht);
      src = e.LayerNorm(e.Binary("Add", src, attn), p + ".norm1");
      std::string ff = e.Gemm(e.Gelu(e.Gemm(src, p + ".linear1.weight", p + ".linear1.bias")),
                              p + ".linear2.weight", p + ".linear2.bias");
      src = e.LayerNorm(e.Binary("Add", src, ff), p + ".norm2");
    }
    proj[static_cast<std::size_t>(nl - 1)] = e.Reshape(e.Transpose(src, {1, 0}), {1, d, ht, ht});
  }

  // --- CCFM: FPN (top-down) then PAN (bottom-up) ---
  std::vector<std::string> inner = {proj[static_cast<std::size_t>(nl - 1)]};
  for (int idx = nl - 1; idx > 0; --idx) {
    const int li = nl - 1 - idx;
    std::string feat_high = e.ConvNorm(inner.front(), fmt::format("lateral_convs.{}", li), 1, 1, true);
    inner[0] = feat_high;
    std::string fused = e.Concat({e.Resize2xNearest(feat_high), proj[static_cast<std::size_t>(idx - 1)]}, 1);
    inner.insert(inner.begin(), e.CSPRep(fused, fmt::format("fpn_blocks.{}", li), 3));
  }
  std::vector<std::string> outs = {inner.front()};
  for (int idx = 0; idx < nl - 1; ++idx) {
    std::string down = e.ConvNorm(outs.back(), fmt::format("downsample_convs.{}", idx), 3, 2, true);
    std::string fused = e.Concat({down, inner[static_cast<std::size_t>(idx + 1)]}, 1);
    outs.push_back(e.CSPRep(fused, fmt::format("pan_blocks.{}", idx), 3));
  }

  // --- decoder_input_proj + flatten -> memory ---
  int Lv = 0;
  std::vector<std::string> mem_flat;
  for (int l = 0; l < nl; ++l) {
    std::string o = e.Bn(e.Conv(outs[static_cast<std::size_t>(l)],
                                fmt::format("decoder_input_proj.{}.0.weight", l), "", 1, 1, 0),
                         fmt::format("decoder_input_proj.{}.1", l));
    const int h = shapes[static_cast<std::size_t>(l)].first;
    const int w = shapes[static_cast<std::size_t>(l)].second;
    mem_flat.push_back(e.Transpose(e.Reshape(o, {d, h * w}), {1, 0}));
    Lv += h * w;
  }
  std::string memory = e.Concat(mem_flat, 0);  // [Lv, d]

  auto heads = e.DeformHead(memory, shapes);
  std::string logits = e.Reshape(heads.first, {1, nq, C});
  std::string boxes = e.Reshape(heads.second, {1, nq, 4});
  g.AddNode("Identity", {logits}, {"logits"});
  g.AddNode("Identity", {boxes}, {"boxes"});
  g.AddOutput("logits", {1, nq, C});
  g.AddOutput("boxes", {1, nq, 4});

  if (e.error) {
    return tl::make_unexpected(*e.error);
  }
  return g.Save(path, /*opset=*/17);
}

core::Result<void> ExportRfDetr(const DetrArch& arch, const weights::StateDict& weights,
                                const std::string& path) {
  GraphBuilder g("rf-detr");
  Emitter e{g, weights, arch, std::nullopt};

  const int d = arch.hidden_dim;
  const int C = arch.num_classes;
  const int nq = arch.num_queries;
  const int nl = arch.num_levels;
  const int hp = arch.imgsz / arch.patch;  // ViT patch grid (square)
  g.AddInput("images", {1, 3, arch.imgsz, arch.imgsz});

  // --- ViT backbone + multi-scale input_proj (1x1 then strided 3x3, GroupNorm) ---
  std::string feat = e.ViTBackbone("images", hp, hp);  // [1, vit_embed, hp, hp]
  std::vector<std::string> proj;
  std::vector<std::pair<int, int>> shapes;
  std::string c0 = e.Conv(feat, "input_proj.0.0.weight", "input_proj.0.0.bias", 1, 1, 0);
  proj.push_back(e.GroupNorm(c0, "input_proj.0.1", 32, d, hp, hp));
  shapes.emplace_back(hp, hp);
  int prev = hp;
  for (int i = 1; i < nl; ++i) {
    std::string c = e.Conv(proj.back(), fmt::format("input_proj.{}.0.weight", i),
                           fmt::format("input_proj.{}.0.bias", i), 3, 2, 1);
    const int sp = (prev - 1) / 2 + 1;
    proj.push_back(e.GroupNorm(c, fmt::format("input_proj.{}.1", i), 32, d, sp, sp));
    shapes.emplace_back(sp, sp);
    prev = sp;
  }

  int Lv = 0;
  std::vector<std::string> mem_flat;
  for (int l = 0; l < nl; ++l) {
    const int h = shapes[static_cast<std::size_t>(l)].first;
    const int w = shapes[static_cast<std::size_t>(l)].second;
    mem_flat.push_back(e.Transpose(e.Reshape(proj[static_cast<std::size_t>(l)], {d, h * w}), {1, 0}));
    Lv += h * w;
  }
  std::string memory = e.Concat(mem_flat, 0);

  auto heads = e.DeformHead(memory, shapes);
  std::string logits = e.Reshape(heads.first, {1, nq, C});
  std::string boxes = e.Reshape(heads.second, {1, nq, 4});
  g.AddNode("Identity", {logits}, {"logits"});
  g.AddNode("Identity", {boxes}, {"boxes"});
  g.AddOutput("logits", {1, nq, C});
  g.AddOutput("boxes", {1, nq, 4});

  if (e.error) {
    return tl::make_unexpected(*e.error);
  }
  return g.Save(path, /*opset=*/17);
}

core::Result<void> ExportDab(const DetrArch& arch, const weights::StateDict& weights,
                             const std::string& path) {
  GraphBuilder g("dab-detr");
  Emitter e{g, weights, arch, std::nullopt};

  const int d = arch.hidden_dim;
  const int half = d / 2;
  const int feat = arch.imgsz / 32;
  const int L = feat * feat;
  const int Q = arch.num_queries;
  const int C = arch.num_classes;

  g.AddInput("images", {1, 3, arch.imgsz, arch.imgsz});

  std::string x = e.Conv(e.ResNet50Backbone("images"), "input_proj.weight", "input_proj.bias", 1, 1, 0);
  std::string src = e.Transpose(e.Reshape(x, {d, L}), {1, 0});  // [L, d]
  e.AddFloatConst("pos", {L, d}, e.SinePos(feat, feat, /*temp=*/20.0));  // DAB temperature 20
  const std::string pos = "pos";

  // --- encoder (PReLU FFN + per-layer encoder_query_scale modulation) ---
  std::string memory = src;
  for (int i = 0; i < arch.enc_layers; ++i) {
    const std::string p = fmt::format("encoder.{}", i);
    std::string scaled_pos = e.Binary("Mul", pos, e.MlpN(memory, "encoder_query_scale", 2));
    std::string q = e.Binary("Add", memory, scaled_pos);
    std::string attn = e.Mha(q, q, memory, p + ".self_attn", L, L);
    memory = e.LayerNorm(e.Binary("Add", memory, attn), p + ".norm1");
    std::string h1 = e.Gemm(memory, p + ".linear1.weight", p + ".linear1.bias");
    std::string ff =
        e.Gemm(e.PRelu(h1, p + ".activation_fn.weight"), p + ".linear2.weight", p + ".linear2.bias");
    memory = e.LayerNorm(e.Binary("Add", memory, ff), p + ".norm2");
  }

  // --- decoder (4D anchors, iterative refinement) ---
  e.AddWeight("refpoint_embed.weight");                                  // [Q, 4]
  std::string reference = e.Unary("Sigmoid", "refpoint_embed.weight");  // [Q, 4]
  e.AddFloatConst("dec_tgt0", {Q, d},
                  std::vector<float>(static_cast<std::size_t>(Q) * static_cast<std::size_t>(d), 0.0F));
  std::string tgt = "dec_tgt0";
  std::string boxes;
  for (int i = 0; i < arch.dec_layers; ++i) {
    const std::string p = fmt::format("decoder.{}", i);
    std::string obj = reference;  // [Q, 4]
    std::string sine4 = e.SineEmbed4DOps(obj, Q);
    std::string query_pos = e.MlpN(sine4, "ref_point_head", 2);  // [Q, d]
    std::string query_sine = e.Slice(sine4, 0, d, 1);            // [Q, d]
    if (i > 0) {
      query_sine = e.Binary("Mul", query_sine, e.MlpN(tgt, "query_scale", 2));
    }
    std::string ref_hw = e.Unary("Sigmoid", e.MlpN(tgt, "ref_anchor_head", 2));  // [Q, 2]
    std::string h_mod = e.Binary("Div", e.Slice(ref_hw, 1, 2, 1), e.Slice(obj, 3, 4, 1));
    std::string w_mod = e.Binary("Div", e.Slice(ref_hw, 0, 1, 1), e.Slice(obj, 2, 3, 1));
    query_sine = e.Concat({e.Binary("Mul", e.Slice(query_sine, 0, half, 1), h_mod),
                           e.Binary("Mul", e.Slice(query_sine, half, d, 1), w_mod)},
                          1);
    tgt = e.CondLayer(p, tgt, memory, pos, query_pos, query_sine, /*is_first=*/i == 0, Q, L,
                      /*prelu=*/true);
    std::string inv_ref = e.InvSigmoid(reference);
    std::string normed = e.LayerNorm(tgt, "decoder_norm");
    boxes = e.Unary("Sigmoid", e.Binary("Add", e.MlpN(normed, "bbox_embed", 3), inv_ref));
    reference = e.Unary("Sigmoid", e.Binary("Add", e.MlpN(tgt, "bbox_embed", 3), inv_ref));
  }

  std::string logits2d =
      e.Gemm(e.LayerNorm(tgt, "decoder_norm"), "class_embed.weight", "class_embed.bias");  // [Q, C]
  std::string logits = e.Reshape(logits2d, {1, Q, C});
  std::string boxes_out = e.Reshape(boxes, {1, Q, 4});
  g.AddNode("Identity", {logits}, {"logits"});
  g.AddNode("Identity", {boxes_out}, {"boxes"});
  g.AddOutput("logits", {1, Q, C});
  g.AddOutput("boxes", {1, Q, 4});

  if (e.error) {
    return tl::make_unexpected(*e.error);
  }
  return g.Save(path, /*opset=*/17);
}

}  // namespace detr::onnxexport
