// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/checkpoint.hpp"

#include <fmt/format.h>

#include <exception>
#include <string>
#include <utility>

#include "detr/weights/safetensors.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::train {

namespace {

using core::Err;
using core::ErrorCode;

void AttachMeta(weights::StateDict& sd, const TrainState& st) {
  sd.SetMeta("format", "detrcpp");
  sd.SetMeta("epoch", std::to_string(st.epoch));
  sd.SetMeta("global_step", std::to_string(st.global_step));
  sd.SetMeta("seed", std::to_string(st.seed));
  sd.SetMeta("best_metric", std::to_string(st.best_metric));
}

TrainState ReadMeta(const weights::StateDict& sd) {
  TrainState st;
  if (auto v = sd.GetMeta("epoch")) {
    st.epoch = std::stoi(*v);
  }
  if (auto v = sd.GetMeta("global_step")) {
    st.global_step = std::stoll(*v);
  }
  if (auto v = sd.GetMeta("seed")) {
    st.seed = std::stoull(*v);
  }
  if (auto v = sd.GetMeta("best_metric")) {
    st.best_metric = std::stod(*v);
  }
  return st;
}

}  // namespace

CheckpointMgr::CheckpointMgr(std::filesystem::path dir) : dir_(std::move(dir)) {
  std::error_code ec;
  std::filesystem::create_directories(dir_, ec);
}

core::Result<void> CheckpointMgr::Save(const std::string& tag, models::IModel& model,
                                       const ModelEma& ema, torch::optim::Optimizer& opt,
                                       const TrainState& state) {
  auto ema_sd = ema.State();
  AttachMeta(ema_sd, state);
  if (auto r = weights::SaveSafetensors(dir_ / (tag + ".safetensors"), ema_sd); !r) {
    return r;
  }

  auto model_sd = weights::StateDictFromModule(model);
  AttachMeta(model_sd, state);
  if (auto r = weights::SaveSafetensors(dir_ / (tag + ".model.safetensors"), model_sd); !r) {
    return r;
  }

  try {
    torch::serialize::OutputArchive archive;
    opt.save(archive);
    archive.save_to((dir_ / (tag + ".opt")).string());
  } catch (const std::exception& e) {
    return Err(ErrorCode::Io, fmt::format("saving optimizer state: {}", e.what()));
  }
  return {};
}

core::Result<TrainState> CheckpointMgr::Load(const std::string& tag, models::IModel& model,
                                             ModelEma& ema, torch::optim::Optimizer& opt) {
  auto model_sd = weights::LoadSafetensors(dir_ / (tag + ".model.safetensors"));
  if (!model_sd) {
    return tl::make_unexpected(model_sd.error());
  }
  auto rep = weights::LoadStateDictInto(model, *model_sd, weights::WeightRemapper{}, false);
  if (!rep) {
    return tl::make_unexpected(rep.error());
  }

  auto ema_sd = weights::LoadSafetensors(dir_ / (tag + ".safetensors"));
  if (!ema_sd) {
    return tl::make_unexpected(ema_sd.error());
  }
  ema.LoadState(*ema_sd);

  try {
    torch::serialize::InputArchive archive;
    archive.load_from((dir_ / (tag + ".opt")).string());
    opt.load(archive);
  } catch (const std::exception& e) {
    return Err(ErrorCode::Io, fmt::format("loading optimizer state: {}", e.what()));
  }

  return ReadMeta(*model_sd);
}

}  // namespace detr::train
