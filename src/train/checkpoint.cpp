// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/train/checkpoint.hpp"

#include <fmt/format.h>

#include <cstring>
#include <exception>
#include <string>
#include <utility>

#include "detr/weights/pth.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/tensor.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::train {

namespace {

using core::Err;
using core::ErrorCode;

// .pth holds a plain dict of tensors (.pth has no side-metadata channel), so the
// train-state scalars ride along as rank-0 tensors under a reserved prefix and are
// dropped before loading into the modules.
constexpr const char* kMetaPrefix = "__detrcpp_meta__.";

weights::RawTensor ScalarI64(std::int64_t v) {
  weights::RawTensor t;
  t.dtype = weights::DType::I64;
  t.data.resize(sizeof(v));
  std::memcpy(t.data.data(), &v, sizeof(v));
  return t;
}
weights::RawTensor ScalarF64(double v) {
  weights::RawTensor t;
  t.dtype = weights::DType::F64;
  t.data.resize(sizeof(v));
  std::memcpy(t.data.data(), &v, sizeof(v));
  return t;
}
template <typename T>
T ReadScalar(const weights::RawTensor* t) {
  T v{};
  if (t != nullptr && t->data.size() >= sizeof(T)) std::memcpy(&v, t->data.data(), sizeof(T));
  return v;
}

void AttachMeta(weights::StateDict& sd, const TrainState& st) {
  sd.Set(std::string(kMetaPrefix) + "epoch", ScalarI64(st.epoch));
  sd.Set(std::string(kMetaPrefix) + "global_step", ScalarI64(st.global_step));
  sd.Set(std::string(kMetaPrefix) + "seed", ScalarI64(static_cast<std::int64_t>(st.seed)));
  sd.Set(std::string(kMetaPrefix) + "best_metric", ScalarF64(st.best_metric));
}

TrainState ReadMeta(const weights::StateDict& sd) {
  TrainState st;
  st.epoch = static_cast<int>(ReadScalar<std::int64_t>(sd.Find(std::string(kMetaPrefix) + "epoch")));
  st.global_step = ReadScalar<std::int64_t>(sd.Find(std::string(kMetaPrefix) + "global_step"));
  st.seed = static_cast<std::uint64_t>(ReadScalar<std::int64_t>(sd.Find(std::string(kMetaPrefix) + "seed")));
  st.best_metric = ReadScalar<double>(sd.Find(std::string(kMetaPrefix) + "best_metric"));
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
  if (auto r = weights::SavePth(dir_ / (tag + ".pth"), ema_sd); !r) {
    return r;
  }

  auto model_sd = weights::StateDictFromModule(model);
  AttachMeta(model_sd, state);
  if (auto r = weights::SavePth(dir_ / (tag + ".model.pth"), model_sd); !r) {
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
  auto model_sd = weights::LoadPth(dir_ / (tag + ".model.pth"));
  if (!model_sd) {
    return tl::make_unexpected(model_sd.error());
  }
  auto rep = weights::LoadStateDictInto(model, *model_sd,
                                        weights::WeightRemapper{}.Drop(kMetaPrefix), false);
  if (!rep) {
    return tl::make_unexpected(rep.error());
  }

  auto ema_sd = weights::LoadPth(dir_ / (tag + ".pth"));
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
