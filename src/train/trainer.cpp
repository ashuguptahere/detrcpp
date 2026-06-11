// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/trainer.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace detr::train {

Trainer::Trainer(std::shared_ptr<models::IModel> model, TrainConfig cfg)
    : model_(std::move(model)),
      cfg_(cfg),
      criterion_(model_->Meta().num_classes, cfg.loss, model_->Meta().focal),
      ema_(*model_, cfg.ema_decay) {
  cfg_.match.focal = model_->Meta().focal;  // matcher cost follows the class mode
  if (cfg_.seed != 0) {
    torch::manual_seed(cfg_.seed);
  }
  // Two parameter groups: the pretrained backbone trains at a lower LR (DETR
  // uses 10x lower), everything else at the base LR. Every model registers its
  // backbone as "backbone", so the name prefix selects it.
  using torch::optim::AdamWOptions;
  using torch::optim::OptimizerParamGroup;
  std::vector<torch::Tensor> backbone_params;
  std::vector<torch::Tensor> head_params;
  for (const auto& p : model_->named_parameters(/*recurse=*/true)) {
    (p.key().starts_with("backbone") ? backbone_params : head_params).push_back(p.value());
  }
  std::vector<OptimizerParamGroup> groups;
  {
    OptimizerParamGroup g(head_params);
    g.set_options(std::make_unique<AdamWOptions>(AdamWOptions(cfg_.lr).weight_decay(cfg_.weight_decay)));
    groups.push_back(std::move(g));
  }
  if (!backbone_params.empty()) {
    OptimizerParamGroup g(backbone_params);
    g.set_options(std::make_unique<AdamWOptions>(
        AdamWOptions(cfg_.lr * cfg_.backbone_lr_mult).weight_decay(cfg_.weight_decay)));
    groups.push_back(std::move(g));
  }
  opt_ = std::make_unique<torch::optim::AdamW>(
      std::move(groups), AdamWOptions(cfg_.lr).weight_decay(cfg_.weight_decay));
}

void Trainer::OnEpochStart(int epoch) {
  if (cfg_.lr_drop <= 0 || epoch != cfg_.lr_drop) {
    return;
  }
  for (auto& group : opt_->param_groups()) {
    auto& opts = static_cast<torch::optim::AdamWOptions&>(group.options());
    opts.lr(opts.lr() * 0.1);
  }
}

float Trainer::TrainStep(const torch::Tensor& images, const TargetBatch& targets) {
  model_->train();
  opt_->zero_grad();

  // DN-DETR: build the denoising group (noised GT queries + group-isolation mask)
  // and run the joint forward; otherwise the plain forward. The matching path
  // below is byte-identical to the non-DN case.
  models::Detections outputs;
  models::DenoisingOut dn_out;
  std::vector<MatchIndices> dn_matches;
  if (model_->SupportsDenoising() && cfg_.dn.dn_number > 0) {
    auto [dn_in, layout] = MakeDenoising(targets, cfg_.dn, model_->Meta().num_classes,
                                         model_->Meta().num_queries);
    if (dn_in.active) {
      dn_in.dn_ref = dn_in.dn_ref.to(images.device());
      dn_in.dn_labels = dn_in.dn_labels.to(images.device());
      dn_in.attn_mask = dn_in.attn_mask.to(images.device());
      dn_matches = BuildDnMatches(layout);
    }
    outputs = model_->ForwardDenoise(images, dn_in, dn_out);
  } else {
    outputs = model_->Forward(images);
  }

  auto matches = HungarianMatch(outputs, targets, cfg_.match);
  auto losses = criterion_.Compute(outputs, targets, matches);
  // Deep supervision: the same set loss on every intermediate decoder layer,
  // each independently matched (DETR's auxiliary losses). Models populate the
  // aux outputs only in training mode; non-deep-supervised models leave them
  // empty and this loop is a no-op.
  for (std::size_t i = 0; i < outputs.aux_logits.size(); ++i) {
    models::Detections aux;
    aux.logits = outputs.aux_logits[i];
    aux.boxes = outputs.aux_boxes[i];
    const auto aux_matches = HungarianMatch(aux, targets, cfg_.match);
    losses.total = losses.total + criterion_.Compute(aux, targets, aux_matches).total;
  }
  // RT-DETRv3 dense supervision: an extra one-to-many loss (each GT supervises its
  // top-k queries) on the final + aux outputs, for denser positive gradient.
  const int o2m_k = model_->DenseSupervisionK();
  if (o2m_k > 0) {
    losses.total =
        losses.total + criterion_.Compute(outputs, targets, OneToManyMatch(outputs, targets, o2m_k,
                                                                            cfg_.match))
                           .total;
    for (std::size_t i = 0; i < outputs.aux_logits.size(); ++i) {
      const models::Detections aux{outputs.aux_logits[i], outputs.aux_boxes[i], {}, {}};
      losses.total =
          losses.total + criterion_.Compute(aux, targets, OneToManyMatch(aux, targets, o2m_k,
                                                                          cfg_.match))
                             .total;
    }
  }

  // DN reconstruction loss: the assignment is KNOWN (no Hungarian) and the same
  // for every decoder layer, so reuse the criterion with the prebuilt matches.
  if (dn_out.active && dn_out.dn_logits.size(1) > 0) {
    const models::Detections d{dn_out.dn_logits, dn_out.dn_boxes, {}, {}};
    losses.total = losses.total + cfg_.dn.weight * criterion_.Compute(d, targets, dn_matches).total;
    for (std::size_t i = 0; i < dn_out.dn_aux_logits.size(); ++i) {
      const models::Detections da{dn_out.dn_aux_logits[i], dn_out.dn_aux_boxes[i], {}, {}};
      losses.total =
          losses.total + cfg_.dn.weight * criterion_.Compute(da, targets, dn_matches).total;
    }
  }
  losses.total.backward();
  if (cfg_.grad_clip > 0.0) {
    torch::nn::utils::clip_grad_norm_(model_->parameters(), cfg_.grad_clip);
  }
  opt_->step();
  ema_.Update(*model_);
  return losses.total.item<float>();
}

}  // namespace detr::train
