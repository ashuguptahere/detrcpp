// Copyright 2026 detrcpp authors. Apache-2.0.
//
// DINOv2 (optionally with-registers) backbone with windowed self-attention — the
// RF-DETR backbone. A faithful port of roboflow/rf-detr's
// dinov2_with_windowed_attn: Conv patch embed + cls token + interpolated position
// embedding, partitioned into num_windows² windows (cls duplicated per window);
// pre-norm transformer blocks with LayerScale; a subset of blocks run global
// (cross-window) attention. Multi-scale features are taken at out_indices and
// un-windowed to [B, C, H, W]. Compiled with DETR_ENABLE_TORCH.

#pragma once

#include <torch/torch.h>

#include <set>
#include <vector>

namespace detr::models {

namespace nn = torch::nn;

// One pre-norm DINOv2 block with LayerScale. Windowed input [B*W², T, C]; when
// run_full it merges windows for the attention then splits back.
struct Dinov2BlockImpl : nn::Module {
  Dinov2BlockImpl(int dim, int heads, int ffn, int num_windows);
  torch::Tensor forward(torch::Tensor x, bool run_full);

  int heads_;
  int num_windows_;
  nn::LayerNorm norm1{nullptr}, norm2{nullptr};
  nn::Linear q{nullptr}, k{nullptr}, v{nullptr}, proj{nullptr};
  nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::Tensor ls1, ls2;  // layer_scale lambda1
};
TORCH_MODULE(Dinov2Block);

struct Dinov2WindowedImpl : nn::Module {
  // pe_grid = sqrt(num patch positions) (e.g. 24 for nano @384/patch16).
  Dinov2WindowedImpl(int embed, int depth, int heads, int patch, int num_windows, int pe_grid,
                     int num_registers, std::vector<int> out_indices,
                     std::vector<int> window_block_indexes);

  // images: [B,3,H,W]. Returns the |out_indices| feature maps, each [B, embed, h, w].
  std::vector<torch::Tensor> forward(torch::Tensor images);

  int embed_, patch_, num_windows_, pe_grid_, num_registers_;
  std::vector<int> out_indices_;
  std::set<int> window_blocks_;

  nn::Conv2d patch_embed{nullptr};
  torch::Tensor cls_token;    // [1,1,C]
  torch::Tensor pos_embed;    // [1, pe_grid²+1, C]
  torch::Tensor reg_tokens;   // [1, num_registers, C] (optional)
  nn::ModuleList blocks{nullptr};
  nn::LayerNorm final_norm{nullptr};  // applied to each out_indices feature
};
TORCH_MODULE(Dinov2Windowed);

}  // namespace detr::models
