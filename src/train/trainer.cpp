// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/trainer.hpp"

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
  auto outputs = model_->Forward(images);
  auto matches = HungarianMatch(outputs, targets, cfg_.match);
  auto losses = criterion_.Compute(outputs, targets, matches);
  losses.total.backward();
  if (cfg_.grad_clip > 0.0) {
    torch::nn::utils::clip_grad_norm_(model_->parameters(), cfg_.grad_clip);
  }
  opt_->step();
  ema_.Update(*model_);
  return losses.total.item<float>();
}

}  // namespace detr::train
