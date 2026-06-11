// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/models/rf_detr_real.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace detr::models {

namespace {

constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

// Per-coordinate sinusoidal embedding (RF-DETR encode_sinusoidal_position_embedding).
// pos: [..., 4] in [0,1]; returns [..., 4*num_feats] with the DETR [y, x, w, h] swap.
torch::Tensor SineEmbed(const torch::Tensor& pos, std::int64_t num_feats) {
  auto dim_t = torch::arange(num_feats, pos.options().dtype(torch::kFloat32));
  dim_t = torch::pow(10000.0, 2 * torch::floor(dim_t / 2) / static_cast<double>(num_feats));
  std::vector<torch::Tensor> embs;
  for (const auto& c : pos.unbind(-1)) {  // x, y, w, h
    auto e = c.unsqueeze(-1) * kTwoPi / dim_t;  // [..., num_feats]
    auto even = e.slice(-1, 0, e.size(-1), 2).sin();
    auto odd = e.slice(-1, 1, e.size(-1), 2).cos();
    embs.push_back(torch::stack({even, odd}, -1).flatten(-2, -1));
  }
  std::swap(embs[0], embs[1]);  // [y, x, w, h]
  return torch::cat(embs, -1);
}

// new = (delta[:2]*ref[2:] + ref[:2], delta[2:].exp()*ref[2:]) — RF-DETR bbox reparam.
torch::Tensor RefineBboxes(const torch::Tensor& ref, const torch::Tensor& delta) {
  auto cxcy = delta.narrow(-1, 0, 2) * ref.narrow(-1, 2, 2) + ref.narrow(-1, 0, 2);
  auto wh = delta.narrow(-1, 2, 2).exp() * ref.narrow(-1, 2, 2);
  return torch::cat({cxcy, wh}, -1);
}

// Grid-anchor proposals for one level: [1, H*W, 4] cxcywh, centers (i+.5)/size, wh 0.05.
torch::Tensor GridProposals(std::int64_t h, std::int64_t w, const torch::TensorOptions& opts) {
  auto gy = torch::arange(h, opts).unsqueeze(1).expand({h, w});
  auto gx = torch::arange(w, opts).unsqueeze(0).expand({h, w});
  auto grid = torch::stack({gx, gy}, -1);  // [h,w,2] (x,y)
  grid = (grid + 0.5) / torch::tensor({static_cast<float>(w), static_cast<float>(h)}, opts);
  auto wh = torch::full({h, w, 2}, 0.05, opts);
  return torch::cat({grid, wh}, -1).view({1, h * w, 4});
}

}  // namespace

RfDecoderLayerImpl::RfDecoderLayerImpl(int d, int sa_heads, int ca_heads, int n_levels,
                                       int n_points, int ffn)
    : sa_heads_(sa_heads) {
  q_proj = register_module("q_proj", nn::Linear(d, d));
  k_proj = register_module("k_proj", nn::Linear(d, d));
  v_proj = register_module("v_proj", nn::Linear(d, d));
  o_proj = register_module("o_proj", nn::Linear(d, d));
  self_attn_layer_norm = register_module("self_attn_layer_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
  cross_attn = register_module("cross_attn", MSDeformAttn(d, n_levels, ca_heads, n_points));
  cross_attn_layer_norm = register_module("cross_attn_layer_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
  fc1 = register_module("fc1", nn::Linear(d, ffn));
  fc2 = register_module("fc2", nn::Linear(ffn, d));
  layer_norm = register_module("layer_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
}

torch::Tensor RfDecoderLayerImpl::forward(torch::Tensor tgt, const torch::Tensor& query_pos,
                                          const torch::Tensor& reference_points,
                                          const torch::Tensor& memory, const SpatialShapes& shapes) {
  const auto b = tgt.size(0);
  const auto t = tgt.size(1);
  const auto c = tgt.size(2);
  const auto hd = c / sa_heads_;
  // Self-attention: q=k=tgt+pos, v=tgt.
  auto qk = tgt + query_pos;
  auto split = [&](const torch::Tensor& y) { return y.view({b, t, sa_heads_, hd}).transpose(1, 2); };
  auto qh = split(q_proj->forward(qk));
  auto kh = split(k_proj->forward(qk));
  auto vh = split(v_proj->forward(tgt));
  auto sc = torch::matmul(qh, kh.transpose(-1, -2)) / std::sqrt(static_cast<double>(hd));
  auto ctx = torch::matmul(sc.softmax(-1), vh).transpose(1, 2).contiguous().view({b, t, c});
  tgt = self_attn_layer_norm->forward(tgt + o_proj->forward(ctx));
  // Deformable cross-attention (query carries the position embedding).
  auto ca = cross_attn->forward(tgt + query_pos, reference_points, memory, shapes);
  tgt = cross_attn_layer_norm->forward(tgt + ca);
  // Residual MLP (decoder_activation_function = relu).
  auto m = tgt + fc2->forward(torch::relu(fc1->forward(tgt)));
  return layer_norm->forward(m);
}

RfDetrRealImpl::RfDetrRealImpl(int num_classes, int num_queries, int imgsz, int vit_embed,
                               int vit_heads, int patch, int num_windows, int dec_layers)
    : num_classes_(num_classes), num_queries_(num_queries), imgsz_(imgsz) {
  const int d = d_model_;
  const int pe = imgsz / patch;  // native pos-embed grid
  backbone_ = register_module(
      "backbone", Dinov2Windowed(vit_embed, 12, vit_heads, patch, num_windows, pe, 0,
                                 std::vector<int>{3, 6, 9, 12},
                                 std::vector<int>{0, 1, 2, 4, 5, 7, 8, 10, 11}));
  projector_ = register_module("projector", RfDetrProjector(4, vit_embed, d, 3));
  enc_output_ = register_module("enc_output", nn::Linear(d, d));
  enc_output_norm_ = register_module("enc_output_norm", nn::LayerNorm(nn::LayerNormOptions({d})));
  enc_out_class_ = register_module("enc_out_class", nn::Linear(d, num_classes));
  enc_out_bbox_ = register_module("enc_out_bbox", Mlp(d, d, 4, 3));
  reference_point_embed_ = register_parameter("reference_point_embed", torch::zeros({num_queries, 4}));
  query_feat_ = register_parameter("query_feat", torch::zeros({num_queries, d}));
  ref_point_head_ = register_module("ref_point_head", Mlp(2 * d, d, d, 2));
  decoder_ = register_module("decoder", nn::ModuleList());
  for (int i = 0; i < dec_layers; ++i) {
    decoder_->push_back(RfDecoderLayer(d, /*sa=*/8, /*ca=*/16, /*levels=*/1, /*points=*/2, 2048));
  }
  dec_layernorm_ = register_module("dec_layernorm", nn::LayerNorm(nn::LayerNormOptions({d})));
  class_embed_ = register_module("class_embed", nn::Linear(d, num_classes));
  bbox_embed_ = register_module("bbox_embed", Mlp(d, d, 4, 3));
}

Detections RfDetrRealImpl::Forward(torch::Tensor images) {
  auto feats = backbone_->forward(images);              // 4 x [B, vit_embed, h, w]
  auto mem_feat = projector_->forward(feats);           // [B, d, h, w]
  const auto b = mem_feat.size(0);
  const auto h = mem_feat.size(2);
  const auto w = mem_feat.size(3);
  auto memory = mem_feat.flatten(2).transpose(1, 2).contiguous();  // [B, hw, d]
  SpatialShapes shapes{{h, w}};

  // Two-stage query selection (group 0).
  auto proposals = GridProposals(h, w, memory.options()).expand({b, -1, -1});  // [B,hw,4]
  auto oq = enc_output_norm_->forward(enc_output_->forward(memory));           // [B,hw,d]
  auto enc_class = enc_out_class_->forward(oq);                                // [B,hw,C]
  auto enc_coord = RefineBboxes(proposals, enc_out_bbox_->forward(oq));        // [B,hw,4]
  auto topk = std::get<1>(std::get<0>(enc_class.max(-1)).topk(num_queries_, 1));  // [B,nq]
  topk_idx_ = topk;
  auto gi = topk.unsqueeze(-1);
  auto topk_coords = enc_coord.gather(1, gi.expand({b, num_queries_, 4}));     // [B,nq,4]
  auto rpe = reference_point_embed_.unsqueeze(0).expand({b, -1, -1});
  auto reference_points = RefineBboxes(topk_coords, rpe);                      // [B,nq,4]
  auto tgt = query_feat_.unsqueeze(0).expand({b, -1, -1}).contiguous();        // [B,nq,d]

  // Decoder (fixed reference points; query position from the sine embed).
  auto ref_input = reference_points.unsqueeze(2);                              // [B,nq,1,4]
  auto query_pos = ref_point_head_->forward(SineEmbed(reference_points, d_model_ / 2));
  for (const auto& layer : *decoder_) {
    tgt = layer->as<RfDecoderLayerImpl>()->forward(tgt, query_pos, ref_input, memory, shapes);
  }
  auto hidden = dec_layernorm_->forward(tgt);

  Detections det;
  det.logits = class_embed_->forward(hidden);                                 // [B,nq,C]
  det.boxes = RefineBboxes(reference_points, bbox_embed_->forward(hidden));    // [B,nq,4] cxcywh
  return det;
}

ModelMeta RfDetrRealImpl::Meta() const {
  ModelMeta m;
  m.name = "rf-detr-nano";
  m.imgsz = imgsz_;
  m.num_classes = num_classes_;
  m.num_queries = num_queries_;
  m.focal = true;            // sigmoid/focal head, no no-object slot
  m.imagenet_norm = true;    // [0,1] then ImageNet mean/std, square resize at imgsz
  m.license = "Apache-2.0";
  m.upstream = "https://github.com/roboflow/rf-detr";
  return m;
}

}  // namespace detr::models
