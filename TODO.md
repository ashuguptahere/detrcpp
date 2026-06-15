# detrcpp — TODO

Master task list, traceable to the approved plan
(`~/.claude/plans/detrcpp-is-the-code-modular-pebble.md`) and the user's full
requirements. `[x]` done, `[~]` in progress, `[ ]` not started. Phases are
ordered for shippability; items inside a phase can parallelize.

> For a concise done/remaining snapshot, see [`docs/STATUS.md`](docs/STATUS.md).
> Highlights: 28 models registered; 3 validated vs official weights (detr-r50
> 41.5, deformable-detr 43.8, conditional-detr 40.3); GPU on Blackwell. Next:
> dab-detr (PReLU + encoder query_scale) and rt-detr (ResNet-VD) validation.

---

## Phase 0 — Skeleton ✅

- [x] Directory tree for all phases (`include/`, `src/`, `apps/`, `bindings/`, …)
- [x] Apache-2.0 `LICENSE`, `.gitignore`, `.clang-format` (Google), `.clang-tidy`
- [x] Top-level `CMakeLists.txt` (C++20, warnings-as-errors, ASan/TSan, ccache)
- [x] `CMakePresets.json` (debug/release/asan/tsan/release-cuda)
- [x] `cmake/dependencies.cmake` — FetchContent deps from source, pinned tags
      (spdlog, fmt, CLI11, tl-expected, simdjson, yaml-cpp, gtest); no package manager
- [x] `cmake/cpu_throttle.cmake` (cores-2 default, Ninja `-l` cap)
- [x] `cmake/license_scan.cmake` (deny GPL/AGPL/LGPL/SSPL/BUSL)
- [x] Centralized `VERSION` → generated `detr/version.hpp` + `bump_version.cmake`
- [x] `detr::core` Result/Error + Device parser
- [x] `detr::log` spdlog façade with NDJSON sink
- [x] `detrcpp` CLI11 wiring (all verbs, long+short forms)
- [x] GoogleTest unit tests + CTest smoke tests
- [x] `.github/workflows/ci.yml` (Linux+macOS, ctest, clang-format, gitleaks)
- [x] `CHANGELOG.md`, `TODO.md`
- [ ] **Apply confirmed fixes from the Phase 0 self-review** (in progress)
- [ ] `git init` + first commit (awaiting user go-ahead)
- [ ] Verify the skeleton actually configures+builds (needs cmake/ninja installed locally)

## Phase 1 — MVP: train + predict + ONNX export

### Data
- [x] `data::Sample` / `data::BBox` core types (normalized cxcywh)
- [x] `data::CocoAdapter` — parse `instances_*.json` via **simdjson** (SIMD)
- [x] `data::YoloAdapter` — `*.txt` labels + `data.yaml` (yaml-cpp)
- [ ] `data::DetrAdapter` — the new single-file **Parquet** format (Apache Arrow)
  - [ ] Schema: `image_path,width,height,boxes(list<struct>),split,fold,hash`
  - [ ] `split` column drives train/val/test selection
  - [x] Seeded deterministic shuffle (cross-platform Fisher–Yates) — done in Dataset
- [ ] `data::Loader` — `std::jthread` pool + lock-free SPSC ring, double-buffered H2D
  - [ ] Image decode via libjpeg-turbo / stb_image
- [x] Format auto-detection (`DetectFormat`) + `LoadDataset` dispatch

### Weight interop (foundational — DONE, proven against LibTorch 2.5.1)
- [x] safetensors reader/writer (Python-free, bidirectional with upstream)
- [x] `StateDict` + `RawTensor` (LibTorch-independent)
- [x] `WeightRemapper` (adapt upstream keys without changing models)
- [x] torch bridge: `StateDictFromModule` / `LoadStateDictInto` (by name + report)
- [ ] Direct `.pth` reader in pure C++ (currently a clear stub → use safetensors)
- [ ] Per-variant remappers validated against official checkpoints

### Models
- [x] LibTorch integration into CMake (`find_package(Torch)`, gated `DETR_ENABLE_TORCH`)
- [x] `models::IModel` interface + `models::Registry` + `RegisterBuiltins()`
- [x] YAML architecture config (hidden_dim/nheads/enc/dec/queries/classes…)
- [x] DETR (conv backbone + transformer enc/dec + queries + class/box heads),
      forward-pass + weight-roundtrip tested, registered, shown in `--list-models`
- [x] Shared transformer head (`detr_head`) — DRY across backbone variants
- [x] **detr-r50** + **detr-r101** (torchvision ResNet backbones) — registered,
      train, ONNX parity-verified; UpstreamRemapper for fb/detr keys
- [x] Official-DETR head alignment (final decoder LayerNorm); parity re-verified
- [x] `.pth` loader (modern zip format) + auto-dispatch; legacy format detected
- [x] **Real COCO mAP, full 5000-img val: mAP50-95 0.415 / mAP50 0.617**
      (official 0.420 / 0.624) via `--coco91 --aspect`; checkpoint converted in
      /tmp (repo stays Python-free)
- [x] Aspect-preserving resize (`--aspect`, batch-1) — closed the small-object AP
      gap (0.129 → 0.163 full-val / 0.207 subset)
- [x] **detr-r50-dc5 / detr-r101-dc5** (dilated C5, stride 16) registered + tested
- [x] **GPU: CUDA LibTorch 2.7.1+cu128 on Blackwell** (`--device cuda:0`, ~17x);
      no-root toolkit via `scripts/setup_cuda_toolkit.sh`
- [ ] Padding mask in the head — enables batched aspect-preserving eval (speed)
- [ ] Legacy (pre-1.6) .pth unpickler in C++ (so no /tmp conversion is needed)
- [ ] Same-`imgsz` enforcement across registered models (default 640)
- [x] **Multi-scale deformable attention op** (grid_sample) — validated vs torch
      (max|Δ|<1e-5); the shared gateway for the deformable family
- [x] **Deformable-DETR** architecture (multi-scale deformable enc/dec + sigmoid
      head) registered + forward-tested
- [x] **Shared focal/sigmoid path** (focal loss + focal matcher cost + sigmoid-
      topk postprocess) — deformable-detr trains on GPU (loss 3639->157/15 ep)
- [x] GPU criterion fix (move matcher indices to the output device)
- [x] **Validated deformable-detr vs official weights → real COCO mAP 43.8**
      (official 44.5); HF convert (rename + qkv concat) loads 0 unexpected
- [x] **RT-DETR** (hybrid AIFI+CCFM encoder + query selection + deformable decoder
      with iterative refinement) — registered, forward + focal-train, GPU-validated
- [ ] RT-DETR depth variants (R18/R34/R101) + validate vs official weights
- [x] **Conditional-DETR** + **DAB-DETR** (shared decoupled decoder layer;
      DAB adds 4D anchors + HW-modulated attn + iterative refinement) — GPU-validated
- [x] **DINO** (deformable encoder + query selection + iterative decoder) — GPU-validated
- [x] **RF-DETR** (ViT backbone + deformable decoder) + a ViT backbone — GPU-validated
- [ ] DN-DETR denoising training (noised GT queries + attn mask) on DAB
- [ ] DINO contrastive denoising (CDN) training
- [ ] RF-DETR DINOv2 backbone fidelity (windowed attn, register tokens) + sizes
- [x] DRY: shared deformable detection head (deform_head) for RT-DETR/RF-DETR/DINO
- [x] **Validated conditional-detr vs official → 40.3** (official 40.9; added decoder_norm)
- [ ] dab-detr official-weight validation needs PReLU activation + encoder query_scale
- [ ] rt-detr official-weight validation needs a ResNet-50-vd backbone (deep stem)
- [ ] Port/verify weights from Apache-2.0 upstreams; document provenance

### Training
- [ ] `train::HungarianMatcher` (Jonker-Volgenant; not in STL/Torch — we implement)
- [ ] `train::DetrLoss` (set-prediction: CE + L1 + GIoU)
- [ ] `train::Trainer` (AdamW, LR scheduler, AMP on CUDA)
- [ ] `train::ModelEma` (decay 0.9999)
- [ ] `train::CheckpointMgr` — write `last.pt` every epoch, `best.pt` on val-mAP gain
- [ ] Resume from `last.pt`: model + optimizer + scheduler + RNG + EMA + epoch + best
- [ ] `--seed` seeds std/torch/CUDA/dataloader RNG; `--deterministic` mode
- [ ] `--export-on-finish onnx` hook

### Inference + Export
- [x] `io::source` — file / directory / glob (url/rtsp/webcam = Phase 3)
- [x] `io::image` (stb load/save/draw) + `infer::preprocess`/`postprocess`
- [x] `detrcpp --predict` — infer, threshold, draw boxes, save annotated PNGs
- [x] `detrcpp --export safetensors` — consolidated weights (Python-free)
- [x] **Hand-written ONNX exporter** (no Python; user decision 2026-06-08):
  - [x] `onnxexport::GraphBuilder` over official onnx lib + `onnx::checker`
  - [x] `ExportDetr(arch, StateDict)` — full DETR graph (MHA decomposed, pos-enc
        constant) mirroring `DetrImpl::Forward`
  - [x] Separate `detrcpp-export` binary (non-torch); main `--export onnx` points to it
  - [x] onnxruntime **parity gate** PASSED (max|Δ| ~6e-7 vs LibTorch)
  - [ ] Generalize ExportDetr per future variant; precision (fp16) + dynamic batch
- [ ] `infer::IBackend` / `TorchBackend` / `OnnxBackend` for `--predict` on .onnx

### Eval
- [x] COCO mAP + **mAP_S / mAP_M / mAP_L** breakdown (small-object metric) —
      faithful pycocotools reimplementation (`detr::eval::CocoEvaluate`), unit-tested
- [x] DETR postprocess (logits/boxes -> absolute-pixel detections, NMS-free)
- [x] `--val` / `--test` wired end-to-end (loads weights, prints the metric table)
- [ ] Per-class AP table; export metrics to JSON for the model-selection graph

### Verification
- [ ] CI integration test: train DETR 1 epoch on 100-img subset, assert loss ↓
- [ ] CI **parity** test: same image through `.pt` vs `.onnx`, IoU>0.99 + class match
- [ ] CI **resume** test: 2+2 epochs == uninterrupted 4 epochs within 1e-6
- [ ] CI **seed** test: same seed → identical batch order + loss

## Phase 2 — More variants + TensorRT + SAHI + trackers

### Model integration roadmap (the full DETR-family target list)

Each model = a `models::IModel` module + a `WeightRemapper`, validated against the
official COCO mAP. **License check (verified 2026-06-14 via the GitHub license API +
the actual LICENSE files): every target below is Apache-2.0 — all clear for our
permissive (Apache-2.0-compatible) policy.** The ONE exception is **Efficient DETR**,
which is *not* a license problem but has **no official public code or weights** (the
Megvii paper was never released), so it cannot be faithfully integrated/validated and
is dropped unless an official checkpoint appears.

Status: ✅ done/validated · 🔄 in progress · ⬜ todo · ⚠️ blocked (no weights) · ❌ no code.
Sizes: families scale either as **N/S/M/L/X** (real-time line) or by **backbone**
(R50/R101/DC5/Swin — the classic line); listed per derivative.

| Model | Status | License | Official repo | Sizes (per derivative) |
|-------|--------|---------|---------------|------------------------|
| **DETR**            | ✅ done (detection) | Apache-2.0 | facebookresearch/detr | R50, R101, R50-DC5, R101-DC5 (+3 *panoptic seg* models — need a mask head, not done) |
| **Deformable-DETR** | ✅ done | Apache-2.0 | fundamentalvision/Deformable-DETR | R50 (+ single-scale / box-refine / two-stage) |
| **Conditional-DETR**| ✅ done | Apache-2.0 | Atten4Vis/ConditionalDETR | R50, R101, R50-DC5, R101-DC5 |
| **Anchor-DETR**     | ⬜ todo | Apache-2.0 | megvii-research/AnchorDETR | R50, R50-DC5 |
| **DAB-DETR**        | ✅ done | Apache-2.0 | IDEA-Research/DAB-DETR | R50, R50-DC5, R101 (+ DAB-Deformable) |
| **DN-DETR**         | ✅ done | Apache-2.0 | IDEA-Research/DN-DETR | R50, R50-DC5 |
| **DINO**            | ✅ done | Apache-2.0 | IDEA-Research/DINO | R50 (4-scale/5-scale), Swin-L |
| **RT-DETR**         | ✅ validated | Apache-2.0 | lyuwenyu/RT-DETR | **S**=R18, **M**=R34, **L**=R50, **X**=R101 (+ our **N**=R18@128) |
| **RT-DETRv2**       | ✅ validated | Apache-2.0 | lyuwenyu/RT-DETR | **S/M/L/X** (R18/R34/R50/R101) |
| **RT-DETRv3**       | ⚠️ blocked | Apache-2.0 | clxia12/RT-DETRv3 | S/M/L/X — **no trained detector weights published** (Paddle; README .pdparams 404s) |
| **LW-DETR**         | ✅ validated | Apache-2.0 | Atten4Vis/LW-DETR | **N**=tiny, **S**=small, **M**=medium, **L**=large, **X**=xlarge (all 5 validated vs native weights) |
| **D-FINE**          | ✅ validated | Apache-2.0 | Peterande/D-FINE | **N/S/M/L/X** + Obj365→COCO `-obj` (all 9 validated, FDR box head) |
| **DEIM**            | ✅ validated | Apache-2.0 | Intellindust-AI-Lab/DEIM | **DEIM-D-FINE n/s/m/l/x** (`deim-*`, SiLU decoder) + **DEIM-RT-DETRv2 s/m/l** (`deim-rt-*`, SiLU + 3-layer query_pos_head) — all validated |
| **DEIMv2**          | 🔄 partial | Apache-2.0 | Intellindust-AI-Lab/DEIMv2 | **N ✅ validated** (`deimv2-n`; new RMSNorm+SwiGLU decoder + sum/CSPLayer2 neck). **atto/femto/pico ✅ validated** (lite encoder + micro HGNetv2; 0.236/0.308/0.383 @ 320/416/640). s/m/l/x (DINOv3-STA backbone) ⬜ |
| **Efficient DETR**  | ❌ no code | (paper-only) | — none — | R50/R101 in paper; **no public code/weights → cannot integrate** |
| **Sparse DETR**     | ⬜ todo | Apache-2.0 | kakaobrain/sparse-detr | R50, Swin-T |
| **Lite DETR**       | ⬜ todo | Apache-2.0 | IDEA-Research/Lite-DETR | R50, Swin-T, Swin-L |
| **Salience-DETR**   | ⬜ todo | Apache-2.0 | xiuqhou/Salience-DETR | R50, Swin-L, FocalNet-L |
| **RF-DETR**         | ✅ nano validated | Apache-2.0 | roboflow/rf-detr | **N/S/M/B/L/X** (nano faithful+validated; S/M/B/L/X = placeholder ViT) |

Priority: classic line [DETR/Deformable/Conditional/DAB/DN/DINO ✅] → RT-DETR(v2) ✅ →
RF-DETR ✅ → LW-DETR ✅ → D-FINE ✅ → DEIM ✅ → **DEIMv2 [next] → Anchor/Sparse/Lite/Salience-DETR**.
(RT-DETRv3 stays blocked on weights; Efficient DETR dropped for lack of code.)

### TensorRT
- [ ] `infer::TrtBackend` (`.engine`)
- [ ] `export::TrtExporter` (+ int8 calibration, fp8, nvfp4)
- [ ] Extend parity test to `.engine` (`.pt` vs `.onnx` vs `.engine`)

### SAHI
- [ ] `sahi::Slicer` (tile size, overlap, pad)
- [ ] `sahi::Stitcher` (cross-tile dedup via bipartite matching, NMS-free)
- [ ] `--sahi` flag on predict/val; report mAP_S improvement

### Trackers (centralized)
- [ ] `track::ITracker` interface + registry
- [ ] Shared single `KalmanFilter` + reuse `HungarianMatcher`
- [ ] SORT
- [ ] ByteTrack
- [ ] OC-SORT
- [ ] BoT-SORT
- [ ] DeepSORT (+ `track::IReid` OSNet embedder, Apache-2.0)
- [ ] NvSORT (gated on NVIDIA SDK)
- [ ] `--track <algo>` flag + YAML config

## Phase 3 — Benchmarks + dataset download

- [ ] `--download <name>` — `configs/datasets.yaml`, libcurl, sha256, resume, ranges
  - [ ] coco2017, voc2012, openimages-v7 entries
  - [ ] auto-convert downloaded set to Parquet format
- [ ] `--benchmark` — measure mAP/FPS per model per device
- [ ] Generate `docs/model_table.md` + `docs/model_table.png` (mAP vs FPS curves)
- [ ] `--list-models` shows name/imgsz/params/license/mAP from release metadata
- [ ] mkdocs docs site; `docs/building.md`, `docs/profiling.md`

## Phase 4 — Edge accelerators (per-vendor, gated on SDK access)

- [ ] Jetson (Nano/Orin/THOR) — reuse TRT path; Jetson preset
- [ ] Axelera backend + exporter
- [ ] MemryX backend + exporter
- [ ] DeepX backend + exporter
- [ ] Hailo backend + exporter
- [ ] Hardware-acceleration probe in `--device auto`

## Phase 5 — Retraining + weight releases

- [ ] `detrcpp --release-weights` (C++ subcommand) — train each variant (`--seed 42`), emit meta.yaml; upload via `gh release upload`
- [ ] Publish `best.pt` + ONNX/TRT to GitHub Releases
- [ ] Regenerate model-selection table from release metadata
- [ ] **Surface compute-budget estimate to user before running** (GPU-hours × $)

## Phase 6 — GUI + mobile (deferred)

- [ ] Clay + SDL3 desktop GUI (`-DDETR_BUILD_GUI=ON`)
  - [ ] Pages: Dataset / Train / Val / Predict / Export
  - [ ] GUI spawns CLI child processes, tails NDJSON logs (no logic divergence)
- [ ] iOS: XCFramework (core/io/infer/track) + CoreML backend (ExecuTorch) + Swift Package
- [ ] Android: NDK `.aar` + NNAPI backend (ExecuTorch)
- [ ] Sample camera apps (`examples/ios`, `examples/android`)

## Cross-cutting (every phase)

- [ ] Tracy zones on hot loops (`-DDETR_TRACY=ON`); `--profile-output`
- [ ] xsimd-vectorized box ops (IoU/GIoU); simdjson for JSON; SIMD image decode
- [ ] Keep deps minimal & centralized (pinned in `cmake/dependencies.cmake`); CI graphs the dep tree
- [ ] clang-tidy + cppcheck gates; gitleaks; no `system()`/shell concat
- [ ] include-what-you-use to prevent bloat/dead code
- [ ] **No Python anywhere** — all tooling is C++ or CMake (`cmake -P` scripts);
      no pybind11, no `.py` helpers. CLI verbs (e.g. `--download`,
      `--release-weights`) replace what would otherwise be Python scripts.
- [ ] Update `CHANGELOG.md` per change; bump `VERSION` per release
