// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/denoising.hpp"

#include <torch/torch.h>

#include <algorithm>

namespace detr::train {

std::pair<models::DenoisingInput, DnLayout> MakeDenoising(const TargetBatch& targets,
                                                          const DnConfig& cfg, int num_classes,
                                                          int num_queries) {
  torch::NoGradGuard ng;
  const int B = static_cast<int>(targets.size());
  const int G = cfg.dn_number;
  int t_pad = 0;
  for (const auto& t : targets) {
    t_pad = std::max<int>(t_pad, static_cast<int>(t.labels.size(0)));
  }
  models::DenoisingInput in;
  DnLayout layout;
  if (t_pad == 0 || G <= 0) {
    return {in, layout};  // active = false
  }
  const int num_dn = G * t_pad;
  const auto fopt = torch::TensorOptions(torch::kFloat32);
  const auto iopt = torch::TensorOptions(torch::kInt64);

  // [B, num_dn, *] buffers with padding defaults (anchor 0.5, label 0, invalid).
  auto dn_ref = torch::full({B, num_dn, 4}, 0.5, fopt);
  auto dn_labels = torch::zeros({B, num_dn}, iopt);
  auto pad_mask = torch::zeros({B, num_dn}, torch::kBool);
  auto tgt_index = torch::full({B, num_dn}, -1, iopt);

  for (int b = 0; b < B; ++b) {
    const int T = static_cast<int>(targets[static_cast<std::size_t>(b)].labels.size(0));
    if (T == 0) {
      continue;
    }
    auto boxes = targets[static_cast<std::size_t>(b)].boxes.to(torch::kCPU).to(torch::kFloat32);  // [T,4] cxcywh
    auto rep = boxes.unsqueeze(0).repeat({G, 1, 1});                    // [G,T,4]
    auto wh = rep.narrow(2, 2, 2);                                      // [G,T,2]
    auto diff = torch::cat({wh * 0.5, wh}, 2);                          // [G,T,4] (center, size)
    auto rand_sign = torch::randint(0, 2, rep.sizes(), fopt) * 2 - 1;   // +-1
    auto rand_part = torch::rand(rep.sizes(), fopt);                    // [0,1)
    auto noised = rep + rand_sign * rand_part * diff * cfg.box_noise_scale;
    noised = noised.clamp(1e-4, 1.0 - 1e-4);  // keep valid sigmoid space

    auto labels = targets[static_cast<std::size_t>(b)].labels.to(torch::kCPU).unsqueeze(0).repeat({G, 1});  // [G,T]
    auto flip = torch::rand({G, T}, fopt) < cfg.label_noise_ratio;
    auto rnd = torch::randint(0, num_classes, {G, T}, iopt);
    labels = torch::where(flip, rnd, labels);  // noised labels (clean kept for the LOSS)

    for (int g = 0; g < G; ++g) {
      const int base = g * t_pad;
      dn_ref[b].narrow(0, base, T) = noised[g];
      dn_labels[b].narrow(0, base, T) = labels[g];
      pad_mask[b].narrow(0, base, T).fill_(true);
      tgt_index[b].narrow(0, base, T) = torch::arange(T, iopt);
    }
  }

  // Group-isolation self-attention mask [L,L] (true = block), batch-shared.
  const int nq = num_queries;
  const int L = num_dn + nq;
  auto mask = torch::zeros({L, L}, torch::kBool);
  mask.narrow(0, num_dn, nq).narrow(1, 0, num_dn).fill_(true);  // matching can't see any dn
  for (int i = 0; i < G; ++i) {
    const int bi = i * t_pad;
    mask.narrow(0, bi, t_pad).narrow(1, num_dn, nq).fill_(true);  // dn group i can't see matching
    for (int j = 0; j < G; ++j) {
      if (j != i) {
        mask.narrow(0, bi, t_pad).narrow(1, j * t_pad, t_pad).fill_(true);  // nor dn group j
      }
    }
  }

  in.active = true;
  in.num_dn = num_dn;
  in.dn_ref = dn_ref.transpose(0, 1).contiguous();        // [num_dn, B, 4]
  in.dn_labels = dn_labels.transpose(0, 1).contiguous();  // [num_dn, B]
  in.attn_mask = mask;
  layout.num_dn = num_dn;
  layout.tgt_index = tgt_index;
  layout.pad_mask = pad_mask;
  return {in, layout};
}

std::vector<MatchIndices> BuildDnMatches(const DnLayout& layout) {
  std::vector<MatchIndices> out;
  const int B = static_cast<int>(layout.pad_mask.size(0));
  out.reserve(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b) {
    auto src = torch::nonzero(layout.pad_mask[b]).squeeze(1);  // valid dn query indices [K]
    auto tgt = layout.tgt_index[b].index_select(0, src);       // the clean GT index [K]
    out.emplace_back(src.to(torch::kInt64), tgt.to(torch::kInt64));
  }
  return out;
}

}  // namespace detr::train
