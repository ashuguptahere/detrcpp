# Changelog

All notable changes to detrcpp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(`MAJOR.MINOR.PATCH`). The authoritative version lives in the top-level `VERSION`
file; use `scripts/bump_version.cmake` (via `cmake -P`) to bump it and promote the
`[Unreleased]` section below.

## [Unreleased]

Phase 1–2 (in progress). Verified against a real toolchain: portable
cmake/ninja/vcpkg + LibTorch (2.5.1 CPU and 2.7.1+cu128 GPU). The full build
(`-DDETR_ENABLE_TORCH=ON`, 60+ tests) passes with zero warnings in project code,
on both CPU and the Blackwell GPU.

### Added
- **GPU support (CUDA).** Builds against CUDA LibTorch (2.7.1+cu128) and runs on
  NVIDIA Blackwell (sm_120) via `--device cuda:0` — ~17x faster than CPU (500-image
  COCO eval: 9s vs ~150s), same numbers. `scripts/setup_cuda_toolkit.sh` assembles
  a CUDA 12.8 build toolkit with no root (pip wheels + NVIDIA redistribs).
- **DETR backbone variants: `detr-r50-dc5`, `detr-r101-dc5`** (dilated C5, output
  stride 16) — five registered ResNet DETR models.
- **Full-val accuracy.** Official detr-r50 weights over all 5000 COCO val images
  (`--aspect`): **mAP50-95 0.415, mAP50 0.617** (official 0.420 / 0.624).
- **Official DETR weights run end-to-end → real COCO mAP.** With the head
  aligned and a COCO-91 eval mode (`--coco91`, raw category ids) plus an
  eval-image cap (`--max-eval`), the official facebookresearch/detr `detr-r50`
  checkpoint loads byte-exact (458 tensors, 0 unexpected) and reproduces the
  published metric: **mAP@[.50:.95] = 0.416 on 500 COCO val images** (official
  full-val 0.420). The legacy-format `.pth` was converted to `.safetensors` by a
  one-off script kept entirely in `/tmp` (the repo stays Python-free). Detections
  on a real COCO image are correct (cat/couch/remote at 0.99+). Remaining gap to
  42.0 is small-object AP from square-resize preprocessing (aspect-preserving
  resize is the tracked follow-up).
- **ResNet-backbone DETR models: `detr-r50` and `detr-r101`** — prove the
  framework is modular end to end. A DRY refactor extracts the shared transformer
  head (`detr_head`) used by every variant; the ResNet builder is parameterized by
  block counts ({3,4,6,3}=R50, {3,4,23,3}=R101). Both register, list, train (a real
  step), predict, evaluate, and **export to ONNX with numeric parity** (the ONNX
  emitter gained a ResNet bottleneck/residual path; max|Δ| ~1e-6 vs LibTorch for
  detr/detr-r50/detr-r101). Configs `detr-r50-tiny.yaml`, `detr-r101-tiny.yaml`.
- **Official-DETR architecture alignment** — added the final decoder LayerNorm
  (`transformer.decoder.norm`) DETR applies before the heads, so our head matches
  facebookresearch/detr structurally (encoder/decoder layers already matched;
  FrozenBatchNorm == eval-mode BN numerically). `UpstreamRemapper` maps the
  official keys (backbone.0.body→backbone, transformer.*.layers→enc/dec,
  transformer.decoder.norm→decoder_norm, multihead_attn→cross_attn,
  bbox_embed.layers.N→…). Parity re-verified after the change.
- **`.pth` loading (`weights::LoadPth`)** — reads modern (zip-format) PyTorch
  checkpoints in pure C++ via LibTorch's own zip reader + unpickler; weight loading
  auto-dispatches `.pth`/`.pt` vs `.safetensors`. Legacy (pre-1.6) checkpoints —
  like the 2020 official DETR weights — are detected and reported clearly (LibTorch
  has no C++ legacy loader; a legacy unpickler is a follow-up).
- **Weight interop (`detr::weights`)** — bidirectional compatibility with the
  original repos, Python-free: `RawTensor`/`DType`, `StateDict`, a
  simdjson-backed safetensors reader/writer, and a `WeightRemapper` that adapts
  upstream parameter names without changing the models. A LibTorch bridge
  (`StateDictFromModule`/`LoadStateDictInto`) makes it concrete; tests prove a
  module's weights round-trip module→safetensors→fresh-module with identical
  values, and that an upstream `model.` prefix is remapped on load.
- **Data adapters (`detr::data`)** — format-independent `Dataset` from a COCO
  adapter (simdjson; crowd-dropping, category remap, xywh→normalized cxcywh),
  a YOLO adapter (yaml-cpp), format auto-detection, and a cross-platform
  seed-reproducible shuffle (Fisher–Yates over mt19937_64). Native Parquet
  "detr" format gated behind the `arrow` vcpkg feature.
- **Model registry + DETR (`detr::models`)** — `IModel`/`ModelMeta`, a
  `Registry` with `RegisterBuiltins()`, and a real DETR model (conv backbone +
  sine positional encoding + multi-head-attention encoder/decoder + object
  queries + class/box heads) that runs a forward pass, trains from scratch, and
  serializes to safetensors. `detrcpp --list-models` lists it in torch builds.
- **Training (`detr::train`)** — the full DETR set-prediction loop:
  `core::LinearSumAssignment` (exact Hungarian, shared with trackers), box ops
  (cxcywh/xyxy/IoU/GIoU), `HungarianMatcher`, `SetCriterion` (CE + L1 + GIoU),
  `ModelEma` (0.9999), `CheckpointMgr` (last/best safetensors + optimizer state),
  and `Trainer` (AdamW + grad clip). A `DataLoader` decodes images (vendored stb)
  and collates batches.
- **`detrcpp --train`** — really trains: builds the model from the registry +
  YAML config, loads a COCO/YOLO dataset, runs epochs with per-epoch checkpoints,
  and `--resume` restores model + EMA + optimizer + epoch. Verified end to end
  (loss decreases across epochs; resume continues the descent).
- **Evaluation + COCO mAP (`detr::eval`)** — `CocoEvaluate` is a faithful,
  LibTorch-free reimplementation of pycocotools COCOeval (10 IoU thresholds,
  101-point interpolation, crowd/ignore/area rules) reporting AP@[.5:.95],
  AP50/75, the **small/medium/large breakdown**, and AR@{1,10,100}; unit-tested
  against hand-computed cases. `infer::PostprocessImage` turns DETR outputs into
  absolute-pixel detections. **`detrcpp --val`/`--test`** loads a checkpoint and
  prints the COCO metric table.
- **Hand-written ONNX export (`detr::onnxexport`)** — Python-free, no torch.onnx:
  `GraphBuilder` over the official ONNX C++ lib (+ `onnx::checker`) and
  `ExportDetr`, which emits the full DETR forward as a fixed-shape ONNX graph
  mirroring `DetrImpl::Forward` (conv backbone, sine positional encoding baked as
  a constant, transformer encoder/decoder with **multi-head attention decomposed**
  into MatMul/Softmax/Transpose/Reshape, class/box heads). Weights are read from
  `.safetensors`, so the exporter needs no LibTorch — it builds as the separate
  `detrcpp-export` binary (gated `DETR_ENABLE_ONNX`, mutually exclusive with
  `DETR_ENABLE_TORCH` because their protobuf providers conflict). **Verified by a
  numeric parity gate**: export → run in onnxruntime → compare to LibTorch =
  max|Δ| 6.6e-7 on logits, 6e-8 on boxes. Tools: `detr-golden` (torch reference),
  `detr-parity` (onnxruntime compare), `scripts/onnx_parity.sh`.
- **Dependencies**: simdjson, yaml-cpp; LibTorch via `find_package(Torch)`
  (gated `DETR_ENABLE_TORCH`); vendored stb image headers (public domain); onnx +
  protobuf and prebuilt onnxruntime (gated `DETR_ENABLE_ONNX`).

### Changed

### Fixed
- `RawTensor::Numel()` now treats an empty shape as a rank-0 scalar (numel 1),
  not numel 0 — so scalar buffers like BatchNorm's `num_batches_tracked`
  serialize correctly (regression-tested).
- `.gitignore`: anchored `data/`/`datasets/` to the repo root so the source
  directories `src/data` and `include/detr/data` are not ignored.

## [0.1.0] - 2026-06-08

Phase 0 — project skeleton. Nothing trains or infers yet; this is the scaffold
every later phase builds on.

### Added
- **Project layout**: full directory tree for all phases (`include/detr/{core,
  data,models,train,infer,track,sahi,export,io,log,util,gui}`, `src/`, `apps/`,
  `bindings/{python,ios,android}`, `tests/`, `benchmarks/`, `configs/`, `docs/`,
  `scripts/`). Later phases drop into existing seams instead of refactoring.
- **Licensing**: Apache-2.0 `LICENSE`. `cmake/license_scan.cmake` denies
  GPL/AGPL/LGPL/SSPL/BUSL in the dependency graph.
- **Build system**: top-level `CMakeLists.txt` (C++20, IPO for release,
  warnings-as-errors for the dangerous subset, opt-in ASan/TSan, ccache/sccache
  auto-wrapping), `CMakePresets.json` (debug/release/asan/tsan/release-cuda),
  `vcpkg.json` manifest (spdlog, fmt, CLI11, tl-expected; gtest/benchmark behind
  features).
- **CPU-safe builds**: `cmake/cpu_throttle.cmake` defaults parallelism to
  `cores - 2` and recommends a Ninja `-l` load cap so the dev machine never
  freezes.
- **Centralized version**: single `VERSION` file → CMake → generated
  `detr/version.hpp` (`detr::Version()`, `DETR_VERSION_*` macros). No hard-coded
  version strings elsewhere. `scripts/bump_version.cmake` (run via `cmake -P`)
  bumps `VERSION`, syncs `vcpkg.json`, and promotes this changelog. The project
  is Python-free: all tooling is C++ or CMake.
- **Core library (`detr::core`)**: `Result<T>`/`Error`/`ErrorCode` (tl::expected
  based), `Device`/`DeviceKind` with `ParseDevice`/`ParseDeviceList` supporting
  `cpu`, `cuda:N`, `mps`, vendor serials, and DDP lists like `cuda:0,1,2,3`.
- **Logging façade (`detr::log`)**: hierarchical loggers over spdlog with an
  optional NDJSON file sink; nothing outside `src/log/` includes spdlog directly.
- **CLI (`detrcpp`)**: CLI11 wiring with long+short flags for
  train/val/test/predict/export/download/benchmark and `--list-models`,
  `--version`, `--help`. Phase-0 subcommands log a structured "not implemented"
  line and return a stable exit code.
- **Tests & CI**: GoogleTest unit tests (device parsing, log registry) + CTest
  smoke tests over the real binary. `.github/workflows/ci.yml` builds
  debug+release on Ubuntu+macOS via vcpkg, runs ctest, enforces clang-format,
  and runs gitleaks.
- **Style**: `.clang-format` (Google, 100 col) and `.clang-tidy`
  (bugprone/cert/cppcoreguidelines/google/modernize/performance/readability).

### Fixed (self-review hardening, pre-release)
- Multi-agent adversarial review of the skeleton; applied all confirmed findings:
  - License scanner had globbed nonexistent `CONTROL` files and always passed —
    rewritten to read the vcpkg SBOM/copyright, allowlist-based (fail-closed),
    fatal on zero metadata.
  - CPU-throttle set a configure-time env var with no effect on the build — now
    generates a real `throttled-build.sh` wrapper (`-j cores-2 -l 0.8·cores`).
  - NDJSON log sink now JSON-escapes message content (was open to log injection
    / invalid JSON lines). Verified: an injected `","k":"v` stays inside `msg`.
  - Device parser rejects out-of-range indices and bare-numeric devices after
    non-indexed kinds (`mps,0` is now an error, not a silent index).
  - CLI is flag-style so the documented `--export=onnx` / `--download=coco2017`
    syntax works; `--log-level` / `--precision` / `--track` are validated.
  - Enum constants renamed to Google style (`Nnapi`, `CoreMl`); CUDA enabled as a
    language only when a compiler is present; CI gains `permissions: contents: read`.

### Removed
- All Python: deleted `bindings/python` (pybind11) and `scripts/bump_version.py`
  (replaced by `scripts/bump_version.cmake`, run via `cmake -P`); dropped the
  `DETR_BUILD_PYTHON` option. The project is Python-free by policy.
