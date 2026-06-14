# CLAUDE.md

Guidance for Claude Code (and human contributors) working in this repository.

## Project

detrcpp — a from-scratch **C++20** DETR-family object-detection framework: train /
eval / predict / export across DETR variants. Backend is **LibTorch** (the PyTorch
C++ API). Permissively licensed (Apache-2.0-compatible deps only). **No Python
anywhere** in the repo — tooling is C++ or CMake `-P` scripts; deps are vendored
from source via CMake **FetchContent** (pinned tags; no package manager). LibTorch
is the one exception (a prebuilt SDK located with `find_package(Torch)`).

## Engineering principles (read first, applies to every change)

These four rules override convenience. When they conflict with "just make it work",
they win.

1. **Think before coding** — state your assumptions out loud, ask when unsure, never
   guess. Reading two files is cheaper than rewriting one.
2. **Simplicity first** — write the minimum code that solves the problem, nothing extra.
   No premature abstractions, no "while-we're-here" refactors.
3. **Surgical changes** — every changed line must trace back to the user's request. If
   you can't justify a hunk in a one-line PR comment, drop it from the diff.
4. **Goal-driven** — turn vague instructions into verifiable success criteria before
   starting. "Validate the model" is not a goal; "official weights load with 0 missing /
   0 unexpected and the COCO mAP lands within ~0.4 of the published figure" is.

### Language + style baseline

- **C++ standard: C++20** (`CMAKE_CXX_STANDARD 20`). The codebase uses few C++20-only
  features; treat C++17 as the de-facto floor and reach for C++20 features only when
  they earn their keep (`std::span` over pointer+size, a `concept` over SFINAE if it
  shortens the call site, etc.).
- **RAII is non-negotiable.** No `new` / `delete` / `malloc` / `free` in `src/` or
  `include/`. Resources go through `std::unique_ptr` / `std::shared_ptr` /
  `std::ifstream` / `std::filesystem` / LibTorch refcounted tensors / the existing
  scope-guard patterns (`torch::NoGradGuard`, `detr::log::Stopwatch`). A new resource
  type gets a custom-deleter `unique_ptr` — never a raw owning pointer.
- **Don't reinvent STL / LibTorch.** Before writing a helper, check `<algorithm>`,
  `<numeric>`, `<ranges>`, `std::filesystem`, `torch::`, and `at::` first. Hand-rolled
  clamp / min / sign / lerp / string-split / file-read utilities are rejected on review.
  The legitimate exceptions are perf-critical or upstream-parity kernels (the legacy
  pickle VM, the MS-deformable-attention sampler, the COCO mAP accumulator) — those stay
  hand-written and are marked as such.
- **SOLID + KISS.** Single-responsibility per file; the model registry + the shared
  heads (`detr_head` / `cond_decoder` / `deform_head`) + the `WeightRemapper` are the
  cross-cutting hooks for per-variant behavior. When a `.cpp` crosses ~800 lines, that's
  a smell — split by responsibility, not by line count.

## Workflow conventions (IMPORTANT)

- **Commit after every fix or feature.** One logical change per commit; do not batch
  unrelated work into one commit. Branch off `main` first — never commit directly to
  `main`. End commit messages with a `Co-Authored-By:` trailer.
- **Versioning is SemVer `MAJOR.MINOR.PATCH`.** The single source of truth is the
  top-level `VERSION` file; never hard-code a version elsewhere. With every change add
  an entry under `## [Unreleased]` in `CHANGELOG.md`. Bump the version when releasing a
  batch of work, using the pure-CMake script (it bumps `VERSION` and promotes
  `[Unreleased]` in `CHANGELOG.md`):
  - `cmake -P scripts/bump_version.cmake -- patch`  — bug / correctness fix
  - `cmake -P scripts/bump_version.cmake -- minor`  — new backward-compatible feature
  - `cmake -P scripts/bump_version.cmake -- major`  — breaking change
- Keep project code warning-clean: the build is `-Wall -Wextra -Wpedantic -Wshadow
  -Wconversion -Wsign-conversion -Wold-style-cast ...`. (Torch-dependent targets
  deliberately do NOT attach the strict `detr_warnings` — third-party headers.)
- Prefer LibTorch / STL / C++20 over hand-rolled code; SIMD-heavy parsing uses
  simdjson. Don't reinvent what the backend already provides.

## Build & test

Requires CMake ≥3.24, Ninja, a C++20 compiler, and LibTorch (no package manager —
the other deps are fetched from source by CMake FetchContent on first configure).
Point CMake at your LibTorch with `-DCMAKE_PREFIX_PATH=/path/to/libtorch`:

```sh
cmake --preset debug -DDETR_BUILD_TESTS=ON -DDETR_ENABLE_TORCH=ON \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build/debug -j
ctest --test-dir build/debug --output-on-failure
```

Omit `-DDETR_ENABLE_TORCH` for the lightweight torch-free build (core/weights/data/
eval/onnx libraries compile without LibTorch). The ONNX exporter
(`-DDETR_ENABLE_ONNX=ON`) is mutually exclusive with torch (protobuf clash).

Validate a model's real COCO mAP (use `--imgsz 800` to match the published DETR
figures; the default 640 scores ~2 AP lower):

```sh
detrcpp --val -m <model> -w <weights> --data <coco-root> --coco91 --aspect [--imgsz 800] [--device cuda:0]
```

## Architecture

- A model implements `models::IModel` (a `torch::nn::Module`) and registers itself in
  `models::RegisterBuiltins()`. Variants share heads: `detr_head` (DETR-backbone
  variants), `cond_decoder` (conditional / dab), `deform_head` (dino / rt-detr /
  rf-detr). Adding a variant = backbone module + reuse a head + one Register line.
- Errors use `tl::expected` (`core::Result<T>`); RAII throughout, no naked `new`.
- Logging goes through the `detr::log` façade over spdlog (NDJSON to disk, with
  timestamp / run-id / thread). Never include `<spdlog/...>` outside `src/log/`.
  Hot-path timing uses `detr::log::Stopwatch`.
- Shared positional encoding lives in `models/pos_embed`; the ResNet backbone uses
  `models::FrozenBatchNorm2d` (eval-identical to BatchNorm-eval, frozen for training).

## Gotchas

- **Weight loading:** always inspect the `LoadReport` (loaded / missing / unexpected /
  mismatched) and refuse to run when `loaded == 0` — otherwise a key-mismatched
  checkpoint runs silently on random weights.
- **Deep supervision:** auxiliary per-decoder-layer losses are emitted by the heads
  only in training mode (empty at inference, so eval/postprocess are untouched).
- **Untrusted inputs:** model files and dataset annotations are parsed defensively
  (safetensors size/offset guards; COCO `file_name` path-traversal rejection).
- Validated-vs-official mAP is measured at `--imgsz 800`.
