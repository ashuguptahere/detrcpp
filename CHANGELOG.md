# Changelog

All notable changes to detrcpp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(`MAJOR.MINOR.PATCH`). The authoritative version lives in the top-level `VERSION`
file; use `scripts/bump_version.py` to bump it and promote the `[Unreleased]`
section below.

## [Unreleased]

Phase 1 (in progress). Verified against a real toolchain: portable
cmake/ninja/vcpkg + LibTorch 2.5.1 CPU. Both the lightweight build
(no LibTorch, 28 tests) and the full build (`-DDETR_ENABLE_TORCH=ON`, 34 tests)
pass with zero warnings in project code.

### Added
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
- **Dependencies**: simdjson, yaml-cpp; LibTorch via `find_package(Torch)`
  (gated `DETR_ENABLE_TORCH`).

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
