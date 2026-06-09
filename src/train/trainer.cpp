// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/trainer.hpp"

#include <memory>
#include <utility>

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
  opt_ = std::make_unique<torch::optim::AdamW>(
      model_->parameters(),
      torch::optim::AdamWOptions(cfg_.lr).weight_decay(cfg_.weight_decay));
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
