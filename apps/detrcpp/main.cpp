// Copyright 2026 detrcpp authors. Apache-2.0.
//
// detrcpp CLI entry point. The verb is selected by a top-level flag/option,
// matching the documented syntax:
//   detrcpp --train     -m rt-detr-l -s 42 [-c cfg] [-d cuda:0] ...
//   detrcpp --val       -m rt-detr-l -w best.pt
//   detrcpp --test      -m rt-detr-l -w best.pt
//   detrcpp --predict   -m rt-detr-l -w best.pt -i <source>
//   detrcpp --export=onnx -m rt-detr-l -w best.pt --precision=fp16
//   detrcpp --download=coco2017 -o data/
//   detrcpp --benchmark | --list-models | --version | --help
//
// Phase 0: parsing, validation, and dispatch are real; the train/val/test/
// predict/export/download bodies log a structured line and return a stable
// "not implemented" exit code. The model registry is still empty.

#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "detr/core/device.hpp"
#include "detr/core/result.hpp"
#include "detr/log/log.hpp"
#include "detr/version.hpp"

#ifdef DETR_ENABLE_TORCH
#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>

#include <torch/torch.h>

#include "detr/data/dataset.hpp"
#include "detr/data/loader.hpp"
#include "detr/data/sample.hpp"
#include "detr/eval/coco_eval.hpp"
#include "detr/eval/evaluator.hpp"
#include "detr/infer/postprocess.hpp"
#include "detr/infer/preprocess.hpp"
#include "detr/io/image.hpp"
#include "detr/io/source.hpp"
#include "detr/models/registry.hpp"
#include "detr/train/checkpoint.hpp"
#include "detr/train/trainer.hpp"
#include "detr/weights/safetensors.hpp"
#include "detr/weights/torch_bridge.hpp"
#endif

namespace {

// Exit codes — kept stable so scripts and the GUI can rely on them.
enum class ExitCode : int {
  Ok = 0,
  UsageError = 2,
  NotImplemented = 64,
  InternalError = 70,
};

// Every option the CLI can set, in one place. CLI11 writes into this directly.
struct Options {
  // Global / logging.
  std::string log_level{"info"};
  std::string log_json;
  std::string log_remote;
  bool quiet{false};

  // Shared across verbs.
  std::string model;
  std::string config;
  std::string data;
  std::string weights;
  std::string source;
  std::string device{"auto"};
  std::optional<std::uint64_t> seed;
  int epochs{12};
  int batch{16};
  int imgsz{640};
  std::string resume;
  bool from_scratch{false};
  bool deterministic{false};
  std::vector<std::string> export_on_finish;

  // Predict.
  std::string track;
  std::string save;
  float conf{0.25F};
  bool sahi{false};
  bool show{false};

  // Export.
  std::string export_format;
  std::string precision{"fp16"};
  std::string calib;

  // Download.
  std::string download_name;
  std::string out{"./data"};

  // Eval.
  bool coco91{false};  // use raw COCO-91 category ids (for official DETR weights)
  int max_eval{0};     // cap eval to N images (0 = all)
};

void ApplyGlobalLogging(const Options& o) {
  static const std::unordered_map<std::string_view, detr::log::Level> kLevels{
      {"trace", detr::log::Level::Trace}, {"debug", detr::log::Level::Debug},
      {"info", detr::log::Level::Info},   {"warn", detr::log::Level::Warn},
      {"error", detr::log::Level::Error},
  };
  if (o.quiet) {
    detr::log::SetGlobalLevel(detr::log::Level::Warn);
  } else {
    const auto it = kLevels.find(o.log_level);
    detr::log::SetGlobalLevel(it != kLevels.end() ? it->second : detr::log::Level::Info);
  }
  if (!o.log_json.empty()) {
    detr::log::EnableJsonSink(o.log_json);
  }
  if (!o.log_remote.empty()) {
    detr::log::EnableRemoteSink(o.log_remote);
  }
}

// Logs a uniform "required option missing" error. Returns false to signal it.
bool Require(bool present, std::string_view flag, std::string_view verb) {
  if (!present) {
    detr::log::Get("cli").error("--{} is required for --{}", flag, verb);
    return false;
  }
  return true;
}

ExitCode RunListModels() {
#ifdef DETR_ENABLE_TORCH
  detr::models::RegisterBuiltins();
  const auto models = detr::models::Registry::Instance().List();
  detr::log::Get("cli").info("registered models: {}", models.size());
  std::cout << fmt::format("{:<14}{:>7}{:>9}{:>9}  {}\n", "name", "imgsz", "queries",
                           "classes", "license");
  for (const auto& m : models) {
    std::cout << fmt::format("{:<14}{:>7}{:>9}{:>9}  {}\n", m.name, m.imgsz, m.num_queries,
                             m.num_classes, m.license);
  }
#else
  detr::log::Get("cli").info("registered models: 0 (built without -DDETR_ENABLE_TORCH)");
  std::cout << "(model registry requires building with -DDETR_ENABLE_TORCH=ON)\n";
#endif
  return ExitCode::Ok;
}

ExitCode NotImplemented(std::string_view verb) {
  detr::log::Get("cli").warn("'{}' is not implemented yet (Phase 0 skeleton)", verb);
  return ExitCode::NotImplemented;
}

#ifdef DETR_ENABLE_TORCH
// Loads weights, dispatching on extension: .pth/.pt go through the pure-C++
// PyTorch reader (official checkpoints), everything else through safetensors.
detr::core::Result<detr::weights::StateDict> LoadWeightsAuto(const std::string& path) {
  const auto ext = std::filesystem::path(path).extension().string();
  if (ext == ".pth" || ext == ".pt") {
    return detr::weights::LoadPth(path);
  }
  return detr::weights::LoadSafetensors(path);
}

torch::Device ToTorchDevice(const detr::core::Device& d) {
  using detr::core::DeviceKind;
  if ((d.kind == DeviceKind::Cuda || d.kind == DeviceKind::Auto) && torch::cuda::is_available()) {
    const int index = d.kind == DeviceKind::Cuda ? d.index : 0;
    return torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(index));
  }
  return torch::Device(torch::kCPU);
}

ExitCode RunTrainTorch(const Options& o, const detr::core::Device& dev) {
  auto& lg = detr::log::Get("cli.train");
  detr::models::RegisterBuiltins();

  YAML::Node cfg;
  if (!o.config.empty()) {
    try {
      cfg = YAML::LoadFile(o.config);
    } catch (const std::exception& e) {
      lg.error("config '{}': {}", o.config, e.what());
      return ExitCode::UsageError;
    }
  }
  if (o.imgsz > 0) {
    cfg["imgsz"] = o.imgsz;
  }

  auto model_r = detr::models::Registry::Instance().Build(o.model, cfg);
  if (!model_r) {
    lg.error("{}", model_r.error().message);
    return ExitCode::UsageError;
  }
  auto model = *model_r;
  const int imgsz = model->Meta().imgsz;

  if (o.data.empty()) {
    lg.error("--data is required for --train");
    return ExitCode::UsageError;
  }
  auto ds_r = detr::data::LoadDataset(o.data);
  if (!ds_r) {
    lg.error("dataset: {}", ds_r.error().message);
    return ExitCode::UsageError;
  }
  auto ds = *ds_r;

  const auto torch_dev = ToTorchDevice(dev);
  model->to(torch_dev);
  lg.info("dataset: {} samples ({} train), {} classes; device {}", ds.Size(),
          ds.CountOf(detr::data::Split::Train), ds.class_names.size(), torch_dev.str());

  const std::uint64_t seed = o.seed ? *o.seed : 0ULL;
  detr::data::DataLoader loader(ds, detr::data::Split::Train, imgsz, o.batch, seed);
  if (loader.NumBatches() == 0) {
    lg.error("no training samples in dataset");
    return ExitCode::UsageError;
  }

  detr::train::TrainConfig tc;
  tc.epochs = o.epochs;
  tc.seed = seed;
  detr::train::Trainer trainer(model, tc);

  const std::filesystem::path run_dir =
      o.resume.empty() ? std::filesystem::path("runs/train") : std::filesystem::path(o.resume);
  detr::train::CheckpointMgr ckpt(run_dir);
  int start_epoch = 0;
  double best = std::numeric_limits<double>::max();
  if (!o.resume.empty()) {
    auto st = ckpt.Load("last", trainer.Model(), trainer.Ema(), trainer.Optimizer());
    if (!st) {
      lg.error("resume from '{}': {}", run_dir.string(), st.error().message);
      return ExitCode::UsageError;
    }
    start_epoch = st->epoch;
    if (st->best_metric > 0) {
      best = st->best_metric;
    }
    lg.info("resumed from epoch {}", start_epoch);
  }

  for (int epoch = start_epoch; epoch < o.epochs; ++epoch) {
    loader.Reshuffle(seed + static_cast<std::uint64_t>(epoch));
    const std::size_t batches = loader.NumBatches();
    double sum = 0.0;
    std::size_t counted = 0;
    for (std::size_t b = 0; b < batches; ++b) {
      auto batch = loader.At(b);
      if (!batch) {
        lg.warn("{}", batch.error().message);
        continue;
      }
      auto images = batch->images.to(torch_dev);
      const float loss = trainer.TrainStep(images, batch->targets);
      sum += static_cast<double>(loss);
      ++counted;
      if (b % 10 == 0) {
        lg.info("epoch {}/{} batch {}/{} loss {:.4f}", epoch + 1, o.epochs, b + 1, batches, loss);
      }
    }
    const double avg = counted ? sum / static_cast<double>(counted) : 0.0;
    lg.info("epoch {}/{} done; avg loss {:.4f}", epoch + 1, o.epochs, avg);

    detr::train::TrainState st;
    st.epoch = epoch + 1;
    st.seed = seed;
    st.best_metric = best;
    if (auto r = ckpt.Save("last", trainer.Model(), trainer.Ema(), trainer.Optimizer(), st); !r) {
      lg.warn("checkpoint save failed: {}", r.error().message);
    }
    if (avg < best) {
      best = avg;
      st.best_metric = best;
      if (auto rb = ckpt.Save("best", trainer.Model(), trainer.Ema(), trainer.Optimizer(), st);
          !rb) {
        lg.warn("best checkpoint save failed: {}", rb.error().message);
      }
      lg.info("new best avg loss {:.4f}", best);
    }
  }
  lg.info("training complete; checkpoints in {}", run_dir.string());
  for (const auto& fmt : o.export_on_finish) {
    lg.warn("--export-on-finish {}: export pipeline lands in a later phase", fmt);
  }
  return ExitCode::Ok;
}
#endif  // DETR_ENABLE_TORCH

ExitCode RunTrain(const Options& o) {
  auto& lg = detr::log::Get("cli.train");
  if (!Require(!o.model.empty(), "model", "train")) {
    return ExitCode::UsageError;
  }
  const auto dev = detr::core::ParseDevice(o.device);
  if (!dev) {
    lg.error("device parse error: {}", dev.error().message);
    return ExitCode::UsageError;
  }
  lg.info("train: model={} device={} epochs={} batch={} imgsz={} seed={} resume={}",
          o.model, detr::core::ToString(*dev), o.epochs, o.batch, o.imgsz,
          o.seed ? std::to_string(*o.seed) : "<random>",
          o.resume.empty() ? "<none>" : o.resume);
#ifdef DETR_ENABLE_TORCH
  try {
    return RunTrainTorch(o, *dev);
  } catch (const std::exception& e) {
    lg.error("training failed: {}", e.what());
    return ExitCode::InternalError;
  }
#else
  lg.error("training requires building with -DDETR_ENABLE_TORCH=ON");
  return NotImplemented("train");
#endif
}

#ifdef DETR_ENABLE_TORCH
ExitCode RunEvalTorch(const Options& o, const detr::core::Device& dev, std::string_view verb,
                      detr::data::Split split) {
  auto& lg = detr::log::Get("cli.eval");
  detr::models::RegisterBuiltins();

  YAML::Node cfg;
  if (!o.config.empty()) {
    try {
      cfg = YAML::LoadFile(o.config);
    } catch (const std::exception& e) {
      lg.error("config '{}': {}", o.config, e.what());
      return ExitCode::UsageError;
    }
  }
  if (o.imgsz > 0) {
    cfg["imgsz"] = o.imgsz;
  }

  auto model_r = detr::models::Registry::Instance().Build(o.model, cfg);
  if (!model_r) {
    lg.error("{}", model_r.error().message);
    return ExitCode::UsageError;
  }
  auto model = *model_r;

  auto sd = LoadWeightsAuto(o.weights);
  if (!sd) {
    lg.error("weights '{}': {}", o.weights, sd.error().message);
    return ExitCode::UsageError;
  }
  auto rep = detr::weights::LoadStateDictInto(*model, *sd, model->UpstreamRemapper(), false);
  if (!rep) {
    lg.error("load weights: {}", rep.error().message);
    return ExitCode::UsageError;
  }
  lg.info("loaded {} tensors ({} missing, {} unexpected)", rep->loaded, rep->missing.size(),
          rep->unexpected.size());

  if (o.data.empty()) {
    lg.error("--data is required for --{}", verb);
    return ExitCode::UsageError;
  }
  auto ds_r = detr::data::LoadDataset(o.data, detr::data::Format::Auto, o.coco91);
  if (!ds_r) {
    lg.error("dataset: {}", ds_r.error().message);
    return ExitCode::UsageError;
  }
  auto ds = *ds_r;

  const auto torch_dev = ToTorchDevice(dev);
  model->to(torch_dev);
  const int imgsz = model->Meta().imgsz;
  lg.info("evaluating {} split: {} samples{}", detr::data::ToString(split), ds.CountOf(split),
          o.max_eval > 0 ? fmt::format(" (capped to {})", o.max_eval) : "");

  const auto m =
      detr::eval::EvaluateModel(*model, ds, split, imgsz, o.batch, torch_dev, o.max_eval);

  auto fmtv = [](double v) { return v < 0 ? std::string(" n/a ") : fmt::format("{:.3f}", v); };
  std::cout << "\nCOCO metrics (" << verb << "):\n";
  std::cout << fmt::format("  mAP @[.50:.95]  : {}\n", fmtv(m.ap));
  std::cout << fmt::format("  mAP @ .50       : {}\n", fmtv(m.ap50));
  std::cout << fmt::format("  mAP @ .75       : {}\n", fmtv(m.ap75));
  std::cout << fmt::format("  mAP  (small)    : {}\n", fmtv(m.ap_small));
  std::cout << fmt::format("  mAP  (medium)   : {}\n", fmtv(m.ap_medium));
  std::cout << fmt::format("  mAP  (large)    : {}\n", fmtv(m.ap_large));
  std::cout << fmt::format("  AR @100         : {}\n", fmtv(m.ar100));
  std::cout << fmt::format("  AR   s / m / l  : {} / {} / {}\n", fmtv(m.ar_small),
                          fmtv(m.ar_medium), fmtv(m.ar_large));
  return ExitCode::Ok;
}
#endif  // DETR_ENABLE_TORCH

ExitCode RunEval(const Options& o, std::string_view verb) {
  auto& lg = detr::log::Get("cli.eval");
  if (!Require(!o.model.empty(), "model", verb) ||
      !Require(!o.weights.empty(), "weights", verb)) {
    return ExitCode::UsageError;
  }
  lg.info("{}: model={} weights={}", verb, o.model, o.weights);
#ifdef DETR_ENABLE_TORCH
  const auto dev = detr::core::ParseDevice(o.device);
  if (!dev) {
    lg.error("device parse error: {}", dev.error().message);
    return ExitCode::UsageError;
  }
  const auto split =
      verb == "test" ? detr::data::Split::Test : detr::data::Split::Val;
  try {
    return RunEvalTorch(o, *dev, verb, split);
  } catch (const std::exception& e) {
    lg.error("eval failed: {}", e.what());
    return ExitCode::InternalError;
  }
#else
  return NotImplemented(verb);
#endif
}

#ifdef DETR_ENABLE_TORCH
// Deterministic per-class color for drawing boxes.
std::array<std::uint8_t, 3> ClassColor(int cls) {
  const auto h = static_cast<unsigned>(cls) * 2654435761U;
  return {static_cast<std::uint8_t>(64 + (h & 0x7F)),
          static_cast<std::uint8_t>(64 + ((h >> 8) & 0x7F)),
          static_cast<std::uint8_t>(64 + ((h >> 16) & 0x7F))};
}

ExitCode RunPredictTorch(const Options& o, const detr::core::Device& dev) {
  auto& lg = detr::log::Get("cli.predict");
  detr::models::RegisterBuiltins();

  YAML::Node cfg;
  if (!o.config.empty()) {
    try {
      cfg = YAML::LoadFile(o.config);
    } catch (const std::exception& e) {
      lg.error("config '{}': {}", o.config, e.what());
      return ExitCode::UsageError;
    }
  }
  if (o.imgsz > 0) {
    cfg["imgsz"] = o.imgsz;
  }

  auto model_r = detr::models::Registry::Instance().Build(o.model, cfg);
  if (!model_r) {
    lg.error("{}", model_r.error().message);
    return ExitCode::UsageError;
  }
  auto model = *model_r;
  auto sd = LoadWeightsAuto(o.weights);
  if (!sd) {
    lg.error("weights '{}': {}", o.weights, sd.error().message);
    return ExitCode::UsageError;
  }
  if (auto rep = detr::weights::LoadStateDictInto(*model, *sd, model->UpstreamRemapper(), false);
      !rep) {
    lg.error("load weights: {}", rep.error().message);
    return ExitCode::UsageError;
  }

  auto sources = detr::io::ResolveImageSources(o.source);
  if (!sources) {
    lg.error("source: {}", sources.error().message);
    return ExitCode::UsageError;
  }

  const auto torch_dev = ToTorchDevice(dev);
  model->to(torch_dev);
  model->eval();
  torch::NoGradGuard no_grad;
  const int imgsz = model->Meta().imgsz;
  const int num_classes = model->Meta().num_classes;

  const std::filesystem::path out_dir = o.save.empty() ? std::filesystem::path("runs/predict")
                                                       : std::filesystem::path(o.save);
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);

  if (o.show) {
    lg.warn("--show needs the GUI (Phase 6); writing annotated images to {} instead",
            out_dir.string());
  }

  std::size_t total = 0;
  for (const auto& path : *sources) {
    auto rgb = detr::io::LoadRgb(path);
    if (!rgb) {
      lg.warn("{}", rgb.error().message);
      continue;
    }
    auto input = detr::infer::PreprocessImage(*rgb, imgsz).to(torch_dev);
    auto outputs = model->Forward(input);
    auto dets = detr::infer::PostprocessImage(outputs, 0, rgb->width, rgb->height, num_classes);
    std::sort(dets.begin(), dets.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    int kept = 0;
    for (const auto& d : dets) {
      if (d.score < o.conf) {
        continue;
      }
      const auto c = ClassColor(d.category_id);
      detr::io::DrawRect(*rgb, static_cast<int>(d.x), static_cast<int>(d.y),
                         static_cast<int>(d.x + d.w), static_cast<int>(d.y + d.h), c[0], c[1],
                         c[2], 2);
      ++kept;
      lg.info("  class={} score={:.3f} box=[{:.0f},{:.0f},{:.0f},{:.0f}]", d.category_id,
              d.score, d.x, d.y, d.w, d.h);
    }
    const auto stem = std::filesystem::path(path).stem().string();
    const auto save_path = out_dir / (stem + "_pred.png");
    if (auto w = detr::io::SavePng(save_path, *rgb); !w) {
      lg.warn("{}", w.error().message);
    }
    lg.info("{}: {} detection(s) >= {:.2f} -> {}", path, kept, o.conf, save_path.string());
    ++total;
  }
  lg.info("predicted {} image(s); results in {}", total, out_dir.string());
  return ExitCode::Ok;
}
#endif  // DETR_ENABLE_TORCH

ExitCode RunPredict(const Options& o) {
  auto& lg = detr::log::Get("cli.predict");
  if (!Require(!o.model.empty(), "model", "predict") ||
      !Require(!o.weights.empty(), "weights", "predict") ||
      !Require(!o.source.empty(), "source", "predict")) {
    return ExitCode::UsageError;
  }
  lg.info("predict: model={} weights={} source={} track={} sahi={}", o.model, o.weights,
          o.source, o.track.empty() ? "<none>" : o.track, o.sahi);
#ifdef DETR_ENABLE_TORCH
  const auto dev = detr::core::ParseDevice(o.device);
  if (!dev) {
    lg.error("device parse error: {}", dev.error().message);
    return ExitCode::UsageError;
  }
  try {
    return RunPredictTorch(o, *dev);
  } catch (const std::exception& e) {
    lg.error("predict failed: {}", e.what());
    return ExitCode::InternalError;
  }
#else
  return NotImplemented("predict");
#endif
}

#ifdef DETR_ENABLE_TORCH
ExitCode RunExportTorch(const Options& o, const detr::core::Device& /*dev*/) {
  auto& lg = detr::log::Get("cli.export");
  detr::models::RegisterBuiltins();

  YAML::Node cfg;
  if (!o.config.empty()) {
    try {
      cfg = YAML::LoadFile(o.config);
    } catch (const std::exception& e) {
      lg.error("config '{}': {}", o.config, e.what());
      return ExitCode::UsageError;
    }
  }
  if (o.imgsz > 0) {
    cfg["imgsz"] = o.imgsz;
  }

  auto model_r = detr::models::Registry::Instance().Build(o.model, cfg);
  if (!model_r) {
    lg.error("{}", model_r.error().message);
    return ExitCode::UsageError;
  }
  auto model = *model_r;
  auto sd_in = LoadWeightsAuto(o.weights);
  if (!sd_in) {
    lg.error("weights '{}': {}", o.weights, sd_in.error().message);
    return ExitCode::UsageError;
  }
  if (auto rep = detr::weights::LoadStateDictInto(*model, *sd_in, model->UpstreamRemapper(), false);
      !rep) {
    lg.error("load weights: {}", rep.error().message);
    return ExitCode::UsageError;
  }

  const std::filesystem::path out_dir =
      o.save.empty() ? std::filesystem::path("runs/export") : std::filesystem::path(o.save);
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);

  // safetensors: a real, supported, Python-free export. Consolidates the loaded
  // checkpoint into a clean weights file the original repos can also load.
  if (o.export_format == "safetensors") {
    auto sd = detr::weights::StateDictFromModule(*model);
    sd.SetMeta("model", o.model);
    sd.SetMeta("imgsz", std::to_string(model->Meta().imgsz));
    const auto path = out_dir / (o.model + ".safetensors");
    if (auto w = detr::weights::SaveSafetensors(path, sd); !w) {
      lg.error("{}", w.error().message);
      return ExitCode::InternalError;
    }
    lg.info("exported {} tensors -> {}", sd.Size(), path.string());
    return ExitCode::Ok;
  }

  // ONNX export runs in the separate torch-free `detrcpp-export` binary: vcpkg's
  // protobuf (for onnx) and LibTorch's bundled protobuf cannot share one link,
  // so the exporter is built without LibTorch and reads weights from safetensors.
  if (o.export_format == "onnx") {
    lg.error("ONNX export runs in the separate torch-free tool `detrcpp-export`.");
    lg.error("Run:  detrcpp-export -m {} {}-w {} -o model.onnx", o.model,
             o.config.empty() ? "" : fmt::format("--config {} ", o.config), o.weights);
    return NotImplemented("export");
  }
  lg.error("export to '{}' is not yet available (onnx is supported via detrcpp-export; "
           "trt/coreml/vendor formats are later phases).",
           o.export_format);
  return NotImplemented("export");
}
#endif  // DETR_ENABLE_TORCH

ExitCode RunExport(const Options& o) {
  auto& lg = detr::log::Get("cli.export");
  if (!Require(!o.model.empty(), "model", "export") ||
      !Require(!o.weights.empty(), "weights", "export")) {
    return ExitCode::UsageError;
  }
  lg.info("export: format={} model={} weights={} precision={}", o.export_format, o.model,
          o.weights, o.precision);
#ifdef DETR_ENABLE_TORCH
  const auto dev = detr::core::ParseDevice(o.device);
  if (!dev) {
    lg.error("device parse error: {}", dev.error().message);
    return ExitCode::UsageError;
  }
  try {
    return RunExportTorch(o, *dev);
  } catch (const std::exception& e) {
    lg.error("export failed: {}", e.what());
    return ExitCode::InternalError;
  }
#else
  return NotImplemented("export");
#endif
}

ExitCode RunDownload(const Options& o) {
  detr::log::Get("cli.download").info("download: name={} out={}", o.download_name, o.out);
  return NotImplemented("download");
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"detrcpp — C++20 DETR-family detection framework", "detrcpp"};
  app.set_version_flag("-v,--version", std::string(detr::Version()));

  Options o;

  // --- Global / logging ---
  app.add_option("--log-level", o.log_level, "log verbosity")
      ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error"}))
      ->capture_default_str();
  app.add_option("--log-json", o.log_json, "NDJSON log file path");
  app.add_option("--log-remote", o.log_remote, "remote log endpoint (http[s]://)");
  app.add_flag("-q,--quiet", o.quiet, "warn-level logging only");

  // --- Verb flags / options (exactly one selects the mode) ---
  bool do_train = false, do_val = false, do_test = false;
  bool do_predict = false, do_benchmark = false, list_models = false;
  app.add_flag("-t,--train", do_train, "train a model");
  app.add_flag("-V,--val", do_val, "validate (split=val)");
  app.add_flag("--test", do_test, "evaluate (split=test)");
  app.add_flag("-p,--predict", do_predict, "run inference on a source");
  app.add_flag("--benchmark", do_benchmark, "run benchmarks and emit the model table");
  app.add_flag("--list-models", list_models, "list registered models and exit");
  auto* export_opt = app.add_option(
      "-x,--export", o.export_format,
      "export to FMT (onnx|trt|coreml|executorch|axelera|memryx|deepx|hailo)");
  auto* download_opt =
      app.add_option("-D,--download", o.download_name, "download a dataset by name");

  // --- Shared options ---
  app.add_option("-m,--model", o.model, "model name (see --list-models)");
  app.add_option("-c,--config", o.config, "YAML architecture/config path");
  app.add_option("--data", o.data, "dataset path");
  app.add_option("-w,--weights", o.weights, "weights path (last.pt / best.pt / .onnx / .engine)");
  app.add_option("-i,--source", o.source,
                 "image | video | glob | url | rtsp://… | webcam:N | -");
  app.add_option("-d,--device", o.device, "cpu | cuda:N | mps | auto | cuda:0,1,…")
      ->capture_default_str();
  app.add_option("-s,--seed", o.seed, "RNG seed (defaults to a random value)");
  app.add_option("-e,--epochs", o.epochs)->capture_default_str();
  app.add_option("-b,--batch", o.batch)->capture_default_str();
  app.add_option("--imgsz", o.imgsz, "image size, shared across all models")
      ->capture_default_str();
  app.add_option("--resume", o.resume, "resume training from a last.pt checkpoint");
  app.add_flag("--from-scratch", o.from_scratch, "ignore pretrained weights");
  app.add_flag("--deterministic", o.deterministic, "deterministic (slower) kernels");
  app.add_option("--export-on-finish", o.export_on_finish,
                 "formats to export after training (e.g. onnx,trt)");

  // Predict-only.
  app.add_option("--track", o.track, "tracker")
      ->check(CLI::IsMember({"sort", "deepsort", "ocsort", "bytetrack", "botsort", "nvsort"}));
  app.add_flag("--sahi", o.sahi, "sliced inference for small objects");
  app.add_option("--save", o.save, "output directory for annotated results");
  app.add_option("--conf", o.conf, "detection confidence threshold")->capture_default_str();
  app.add_flag("--show", o.show, "display results");

  // Export-only.
  app.add_option("--precision", o.precision, "fp32|fp16|bf16|int8|int4|fp8|nvfp4")
      ->check(CLI::IsMember({"fp32", "fp16", "bf16", "int8", "int4", "fp8", "nvfp4"}))
      ->capture_default_str();
  app.add_option("--calib", o.calib, "int8 calibration directory");

  // Download-only.
  app.add_option("-o,--out", o.out, "download output directory")->capture_default_str();

  // Eval-only.
  app.add_flag("--coco91", o.coco91, "use raw COCO-91 category ids (official DETR weights)");
  app.add_option("--max-eval", o.max_eval, "cap evaluation to N images (0 = all)");

  CLI11_PARSE(app, argc, argv);

  try {
    ApplyGlobalLogging(o);

    if (list_models) {
      return static_cast<int>(RunListModels());
    }

    // Exactly one verb may be active.
    const bool do_export = export_opt->count() > 0;
    const bool do_download = download_opt->count() > 0;
    const int verb_count = static_cast<int>(do_train) + static_cast<int>(do_val) +
                           static_cast<int>(do_test) + static_cast<int>(do_predict) +
                           static_cast<int>(do_benchmark) + static_cast<int>(do_export) +
                           static_cast<int>(do_download);
    if (verb_count == 0) {
      std::cout << app.help() << '\n';
      return static_cast<int>(ExitCode::Ok);
    }
    if (verb_count > 1) {
      detr::log::Get("cli").error(
          "choose exactly one mode "
          "(--train/--val/--test/--predict/--export/--download/--benchmark)");
      return static_cast<int>(ExitCode::UsageError);
    }

    if (do_train) {
      return static_cast<int>(RunTrain(o));
    }
    if (do_val) {
      return static_cast<int>(RunEval(o, "val"));
    }
    if (do_test) {
      return static_cast<int>(RunEval(o, "test"));
    }
    if (do_predict) {
      return static_cast<int>(RunPredict(o));
    }
    if (do_export) {
      return static_cast<int>(RunExport(o));
    }
    if (do_download) {
      return static_cast<int>(RunDownload(o));
    }
    if (do_benchmark) {
      return static_cast<int>(NotImplemented("benchmark"));
    }
    return static_cast<int>(ExitCode::Ok);
  } catch (const std::exception& e) {
    detr::log::Get("cli").critical("uncaught exception: {}", e.what());
    return static_cast<int>(ExitCode::InternalError);
  }
}
