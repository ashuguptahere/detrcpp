// Copyright 2026 detrcpp authors. Apache-2.0.
//
// The LW-DETR ViT backbone (Atten4Vis/LW-DETR): a plain windowed ViT (CAEv2-style)
// with interleaved windowed/global attention and LayerScale, producing several
// intermediate feature maps for the C2f projector. Distinct from the DINOv2-windowed
// backbone (no cls token, bicubic absolute pos-embed, separate q/k/v with bias-free
// k, no final layernorm on the emitted features). Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include <set>
#include <vector>

namespace detr::models {

namespace nn = torch::nn;

// One LW-DETR ViT block: pre-norm self-attention (separate q/k/v/o, k has no bias) +
// LayerScale gamma_1, then a pre-norm GELU MLP + LayerScale gamma_2. Windowed blocks
// attend within each window; global blocks merge the windows first (run_full).
struct LwDetrViTBlockImpl : nn::Module {
  LwDetrViTBlockImpl(int dim, int heads, int ffn, int num_windows_side);
  torch::Tensor forward(torch::Tensor x, bool run_full);

  int heads_;
  int nws_;  // windows per side (num_windows = nws_^2)
  nn::LayerNorm norm1{nullptr}, norm2{nullptr};  // layernorm_before / after
  nn::Linear q{nullptr}, k{nullptr}, v{nullptr}, o{nullptr};
  nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::Tensor gamma1, gamma2;
};
TORCH_MODULE(LwDetrViTBlock);

// The backbone: conv patch-embed + bicubic absolute pos-embed, window-partition into
// nws^2 windows, run `depth` blocks (windowed except the global-block indexes), and
// emit the un-windowed feature maps at the requested layer indexes.
class LwDetrViTImpl : public nn::Module {
 public:
  LwDetrViTImpl(int embed, int depth, int heads, int patch, int num_windows_side, int pe_grid,
                std::vector<int> out_layer_indexes, std::vector<int> window_block_indexes);

  // images: [B, 3, H, W] -> N feature maps, each [B, embed, H/patch, W/patch].
  std::vector<torch::Tensor> forward(torch::Tensor images);

 private:
  int embed_, patch_, nws_, pe_grid_;
  std::vector<int> out_layers_;
  std::set<int> window_blocks_;
  nn::Conv2d patch_embed{nullptr};
  torch::Tensor pos_embed;  // [1, pe_grid^2 + 1, embed] (cls slot dropped at use)
  nn::ModuleList blocks{nullptr};
};
TORCH_MODULE(LwDetrViT);

}  // namespace detr::models
