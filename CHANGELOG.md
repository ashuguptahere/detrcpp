# Changelog

All notable changes to detrcpp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(`MAJOR.MINOR.PATCH`). The authoritative version lives in the top-level `VERSION`
file; use `scripts/bump_version.cmake` (via `cmake -P`) to bump it and promote the
`[Unreleased]` section below.

## [Unreleased]

### Added

### Changed

### Fixed

## [0.13.1] - 2026-06-10

### Added

### Changed

### Fixed
- **Legacy `.pth` unpickler now loads real-world checkpoints.** The initial version
  only handled the synthetic round-trip; testing against a real legacy DETR
  checkpoint surfaced constructs torch actually emits, now all supported: Python-2
  `str` keys (`SHORT_BINSTRING`/`BINSTRING`), integer storage ids (normalized to
  their decimal string to match the storage-key list), the `_rebuild_tensor` (v1)
  global, and — for training checkpoints — `BINFLOAT`/`NEWOBJ` (optimizer floats /
  argparse namespaces, parsed as inert) plus unwrapping a state_dict nested under
  `model`/`state_dict`. Per-storage dtypes are now recorded at every `persistent_id`
  (so the storage walk never lacks a size, even for shared/optimizer storages),
  making an unrecognized file fail cleanly with `Unsupported` instead of erroring
  mid-walk. **Verified end-to-end:** loading the real legacy `detr-r50` checkpoint
  reports 458 tensors, 0 missing / 0 unexpected / 0 shape-mismatched, and the model
  detects correctly (cats @ 0.998). A gated `DETR_LEGACY_PTH` test exercises a real
  file when present (CI-skipped otherwise).

## [0.13.0] - 2026-06-10

### Added
- **RT-DETR-CDN training recipe.** A new `rt-detr-cdn` model — RT-DETR (ResNet-VD +
  hybrid AIFI/CCFM encoder + shared deformable head) with contrastive denoising
  training, reusing the shared deform-head CDN entry + the contrastive noise
  generator (a `denoising` flag + `label_enc` + an `Encode`/`Forward` split +
  `ForwardDenoise`, mirroring dino-cdn/rf-detr-cdn; no new infra). A single
  training-recipe alias (the -l config), not a size matrix. Inference is plain
  RT-DETR; rt-detr/v2/v3 stay byte-identical. Verified by an overfit test. This
  brings CDN denoising to every deformable family (dino, rf-detr, rt-detr).
- **RT-DETRv2 discrete sampling.** A round-to-nearest variant of multi-scale
  deformable attention (`MSDeformAttnCore(..., discrete=true)`): instead of
  bilinear-interpolating the value at the continuous sampling location, it rounds
  to the nearest pixel and gathers — a deployment-friendly op. Threaded through
  `MSDeformAttn` / `BuildDeformDetectHead` as a default-`false` flag and a
  `discrete_sample` config key, enabled only for the **rt-detrv2** matrix; bilinear
  (rt-detr / rt-detrv3 / dino / rf-detr / deformable-detr) stays byte-identical.
  Inference-affecting (it's the op, not a training branch). Verified that v2 differs
  from v1 while v3 matches v1 at equal weights.
- **ByteTrack multi-object tracker** (`detr::track`, new torch-free `detr_track`
  lib). An 8-dim constant-velocity Kalman filter (height-scaled noise, a 4×4 SPD
  Cholesky solve in the update) per track + two-stage IoU association — high-score
  detections first, then low-score detections against the leftovers — reusing
  `core::LinearSumAssignment`, with a coast-then-reap lifecycle (lost tracks coast
  up to `track_buffer` frames, re-associating on reappearance). Boxes are
  absolute-pixel xyxy; the IoU formula matches `box_ops::BoxIou`. Builds in the
  lightweight (no-LibTorch) config. Tested: stable id on a moving box, id
  persistence through an occlusion gap, stage-2 low-score recovery, the spawn gate,
  buffer expiry, and Kalman convergence.
- **SAHI sliced inference** (`detr::infer::SahiDetect`, `--sahi`). Slices a large
  image into overlapping tiles (`ComputeSlices`, last tile flush to the border),
  detects per tile, shifts each tile's boxes back to full-image coordinates, and
  merges across tiles with the repo's first NMS (`NmsPerClass`, class-aware, built
  on `train::BoxIou`; an optional NMM mode folds seam-straddling boxes). A
  `TileDetector` callback decouples the tiling from the model (a model overload
  runs preprocess→Forward→postprocess on the model's device); output is the same
  `DtBox` xywh as the single-pass path, so the predict draw/save tail is reused via
  one `--sahi` branch. Improves small-object recall on high-resolution inputs.
- **Legacy (pre-1.6) `.pth` unpickler** (`weights::LoadLegacyPth`, pure C++). Old
  `torch.save` checkpoints are a raw protocol-2 pickle of an `OrderedDict[str ->
  Tensor]` followed by length-prefixed little-endian storages — a format LibTorch's
  C++ API cannot read. A minimal pickle VM (~30 opcodes) recognizes only the globals
  a state_dict emits (`_rebuild_tensor_v2`, `_rebuild_parameter`, `OrderedDict`, the
  storage classes), never dispatches an unknown global (no code execution; unknown
  reduces become inert), and walks the storage section to slice each tensor's bytes
  into a `RawTensor`. It returns **Unsupported on anything exotic** (unknown global,
  big-endian, non-contiguous stride) rather than a wrong tensor — strictly better
  than the previous blanket rejection, and the `loaded==0` guard remains the safety
  net. Wired into the existing `0x80` legacy-magic branch in `LoadPth`, so old `.pth`
  files now load with no CLI change. Torch-free (`detr_weights`), so it builds in the
  lightweight config; tested via an in-test legacy writer round-trip + truncation /
  unknown-global / big-endian adversarial cases.

### Changed

### Fixed

## [0.12.1] - 2026-06-10

### Added

### Changed

### Fixed
- **Deterministic overfit tests.** The denoising / dense-supervision overfit-a-tiny-
  batch tests (dn-detr, dino-cdn, rf-detr-cdn, rt-detrv3) seeded via `tc.seed=0`,
  which is the "no seed" sentinel — so model init + the random batch were unseeded
  and the loss-threshold assert was flaky under parallel CI (it tripped once for
  rt-detrv3, whose larger dense-supervision loss converges slower). They now call
  `torch::manual_seed(0)` before building the model, making each trajectory
  reproducible; rt-detrv3's loop is lengthened to 60 steps to clear the threshold
  with margin.

## [0.12.0] - 2026-06-10

### Added
- **RT-DETRv3 dense supervision training recipe.** Hierarchical dense positive
  supervision: a new `OneToManyMatch` assigns each ground-truth its top-k
  lowest-cost queries (reusing the Hungarian cost matrix), and the trainer adds
  that one-to-many loss (on the final + aux outputs) for denser positive gradient.
  Gated by a new `IModel::DenseSupervisionK()` virtual — set to k=6 only for the
  `rt-detrv3` registry matrix (rt-detr / rt-detrv2 stay at 0, so they're
  unchanged). Train-only; inference is unaffected. Verified by a one-to-many
  property test, the per-version gating, a training step, and an overfit test.

### Changed

### Fixed

## [0.11.0] - 2026-06-10

### Added
- **DINO-CDN (contrastive denoising) training.** A new `dino-cdn` model (the DINO
  network plus a `label_enc`) that extends the DN-DETR denoising infra with
  **contrastive** queries: per GT, a *positive* (small box noise) query that
  reconstructs the GT and a *negative* (large box noise, `rand_part ∈ [1,2)`) query
  that must predict **no-object** — taught to reject low-quality anchors. The
  contrastive layout (`DnConfig.contrastive`) packs each group as `[positive_T |
  negative_T]` with stride `2·T`; `BuildDnMatches` matches only positives, so the
  criterion gives negatives the background loss (no box loss). The shared deform
  head gained an optional CDN entry (prepends the denoising queries + an additive
  self-attention mask, splits the outputs) and the deform decoder layer an optional
  self-attn mask — both default-off, so rt-detr/rf-detr/dino-plain/deformable-detr
  stay byte-identical (regression-verified). Inference is plain DINO. Verified by a
  CDN-layout property test + an overfit-a-tiny-batch test.
- **RF-DETR-CDN training recipe.** A new `rf-detr-cdn` model — RF-DETR (ViT backbone)
  with contrastive denoising training, reusing the shared deformable head's CDN entry
  and the contrastive noise generator wholesale (just a `denoising` flag + `label_enc`
  + `ForwardDenoise`, mirroring dino-cdn; no new infra). Inference is plain RF-DETR;
  plain rf-detr byte-identical. Completes the denoising-training trilogy
  (DN-DETR → DINO-CDN → RF-DETR-CDN). Verified by an overfit-a-tiny-batch test.

### Changed

### Fixed

## [0.10.0] - 2026-06-10

### Added
- **DN-DETR denoising training.** A new `dn-detr` model (the DAB-DETR network plus
  a `label_enc`) and a train-only denoising recipe: per image, `dn_number` groups
  of denoising queries whose anchor is a noised GT box and whose content embeds a
  noised GT label are run through the decoder alongside the matching queries, with
  a group-isolation self-attention mask (matching ↛ dn, dn group i ↛ group j ↛
  matching). The denoising part trains with a **reconstruction loss against the
  clean GT** (a known assignment, no Hungarian — reuses `SetCriterion`); the
  matching part is unchanged. New `train/denoising.{hpp,cpp}` (noise + mask +
  known `MatchIndices`); `IModel` gained tensor-only `DenoisingInput`/
  `DenoisingOut` + `SupportsDenoising()`/`ForwardDenoise()` (default no-ops);
  `DecoupledMultiHeadAttn`/`CondDecoderLayer` gained an optional self-attn mask
  (default none → conditional/dab byte-identical). Inference is the plain DAB
  forward (denoising is train-only). Verified by an overfit-a-tiny-batch test.

### Changed

### Fixed

## [0.9.0] - 2026-06-10

### Added
- **ONNX export for rf-detr** — completes ONNX export for the whole model zoo.
  `ExportRfDetr` reconstructs the **ViT backbone** in pure ONNX (patch-embed conv +
  a 2D sin-cos position + pre-norm transformer blocks with a GELU FFN + a final
  LayerNorm), then a multi-scale GroupNorm projection and the shared topk deform
  head. The decomposed `Mha` gained dim/head overrides so the ViT can run at
  `vit_embed`/`vit_heads` independent of the head dim. Verified at **max|Δ| ≈ 2e-6
  vs LibTorch** (`configs/models/rf-detr-tiny.yaml`, with vit_embed≠hidden_dim).
  All nine model families now export to ONNX (~5e-7–4e-6 parity).

### Changed

### Fixed

## [0.8.0] - 2026-06-10

### Added
- **ONNX export for rt-detr** (the real-time model). `ExportRtDetr` reconstructs the
  whole stack in pure ONNX: a **ResNet-D/VD backbone** (deep 3×(3×3) stem + AvgPool
  downsample shortcuts), the **hybrid encoder** (an AIFI transformer on the top level
  with a GELU FFN — emitted via `Erf` — and a 2D sin-cos position, then a **CCFM**
  conv FPN/PAN of ConvNorm/RepVgg/CSPRep with nearest-2× `Resize`), a
  `decoder_input_proj`, and the shared topk deformable head. New emit helpers:
  AvgPool, Resize, SiLU, GELU, ConvNorm/RepVgg/CSPRep, ResNet-VD stages, SinCos2d.
  Verified at **max|Δ| ≈ 7e-7 vs LibTorch** (`configs/models/rt-detr-tiny.yaml`).

### Changed

### Fixed

## [0.7.0] - 2026-06-10

### Added
- **ONNX export for dino** — completes the deformable query-selection path. Reuses
  the deformable encoder (MSDeformAttn) and adds the shared **deform head**: grid-
  center anchors baked as a constant, **topk query selection** (ReduceMax → TopK →
  Gather over the encoder classification), and a 4D-reference deformable decoder
  with iterative box refinement. New emit helpers: ReduceMax, TopK, Gather, Expand.
  Verified at **max|Δ| ≈ 4e-6 vs LibTorch** (`configs/models/dino-tiny.yaml`). Six
  deformable models now share the GridSample MSDeformAttn + topk head (rt-detr/rf
  remain, needing ResNet-VD+CCFM / ViT backbone emit).

### Changed

### Fixed

## [0.6.0] - 2026-06-10

### Added
- **ONNX export for deformable-detr** — the deformable tier. A hand-written
  `ExportDeformable` reconstructs **multi-scale deformable attention** in pure ONNX:
  per-level **GridSample** (bilinear, zeros-pad, align_corners=0) at `2·loc−1`,
  weighted by the softmaxed attention, in both the 6-layer deformable encoder and
  decoder. Adds a manual **GroupNorm** (via `InstanceNormalization`, opset-17-safe)
  for the 4-level input projection, a multi-tap `ResNet50Stages` (C3/C4/C5), and
  bakes the encoder pos/reference grid + fixed query reference as constants.
  Verified at **max|Δ| ≈ 8e-7 vs LibTorch** (`configs/models/deformable-tiny.yaml`);
  the GridSample MSDeformAttn helper is reusable for dino/rt-detr.

### Changed

### Fixed

## [0.5.0] - 2026-06-10

### Added
- **ONNX export for conditional-detr** (focal). A new hand-written `ExportConditional`
  mirrors the conditional decoder op-for-op: decoupled self-attention, conditional
  cross-attention (query content + sine concatenated head-wise), and the fixed sine
  reference (precomputed exactly from the learned query embeddings). Verified at
  **max|Δ| ≈ 5e-7 vs LibTorch** via the parity harness (`configs/models/conditional-
  tiny.yaml`). Wired into `detrcpp-export` and `detr-parity`.
- **ONNX export for dab-detr** (focal). `ExportDab` mirrors the DAB decoder
  op-for-op including the parts conditional doesn't have: 4D dynamic anchors with
  iterative box refinement (`bbox_embed` on the LayerNorm'd state for the box, on
  the raw state for the next reference), the `SineEmbed4D` interleave emitted as
  dynamic Slice/Sin/Cos ops, width/height-modulated sine queries, PReLU FFNs, the
  per-layer encoder `query_scale` modulation, and sine temperature 20. Verified at
  **max|Δ| ≈ 5e-7 vs LibTorch** (`configs/models/dab-tiny.yaml`).

### Changed
- ONNX emit: shared weights (per-layer-applied MLPs like `bbox_embed`,
  `encoder_query_scale`) are added as an initializer once; the parity runner runs
  the graph with optimizations disabled (purest check + sidesteps an ORT fusion
  bug on Slice-heavy graphs).

### Fixed

## [0.4.0] - 2026-06-10

### Added
- **rt-detr validated against official weights** — the HF PekingU/rtdetr_r50vd
  checkpoint loads 0-missing/0-unexpected and reproduces real COCO mAP
  (**mAP50-95 0.530** on full val; official 0.534), the 5th model validated and
  the first with no debug iterations. Adds a **ResNet-D/VD backbone** (gated
  `deep_stem` + `avg_down` on `ResNetImpl`, default off so every other model is
  byte-identical), drops CSPRep's `conv3` (HF's is Identity), switches the AIFI
  FFN to GELU, and adds `decoder_input_proj`.
- `ModelMeta::imagenet_norm` (default true) + a `normalize` flag on
  `PreprocessImage`: RT-DETR runs on raw [0,1] inputs with a square (non-aspect)
  eval resize. DETR-family preprocessing is unchanged.

### Changed

### Fixed

## [0.3.0] - 2026-06-10

### Added
- **dab-detr validated against official weights** — the HF
  IDEA-Research/dab-detr-resnet-50 checkpoint loads 0-missing/0-unexpected and
  reproduces real COCO mAP (**mAP50-95 0.419** on full val; official 0.409), the
  4th model validated end-to-end. Aligned the architecture with official DAB:
  PReLU FFN (encoder + decoder), an encoder `query_scale`, a decoder LayerNorm,
  the bbox-head-on-normed-state split, and **sine temperature 20** (DAB's
  signature, vs DETR's 10000). `SinePos` gained a `temperature` param (default
  10000), so detr/r50/r101/dc5/dino are byte-identical; conditional-detr's
  shared decoder layer stays relu by default (regression-verified).

## [0.2.5] - 2026-06-10

### Changed
- **Logging façade no longer leaks spdlog.** `detr/log/log.hpp` is spdlog-free
  (pulls the lighter `<fmt/format.h>`); `Get()` returns a `Logger` handle whose
  format methods are compile-time-checked and forward to a non-template sink, so
  spdlog is confined to `src/log/`. No behavior change (human + NDJSON output and
  escaping unchanged).

## [0.2.4] - 2026-06-10

### Added
- **`--device` downgrade warning.** An explicit CUDA request that can't be
  honored (no CUDA), or any device kind this build cannot target (mps/vulkan/
  coreml/hailo/...), now logs a warning instead of silently using the CPU.

## [0.2.3] - 2026-06-10

### Added
- **Inference throughput in `predict` and `val`/`test`.** `predict` logs aggregate
  latency + FPS (`predicted N image(s) in T ms inference (X img/s, Y ms/img)`),
  timed around preprocess→forward→postprocess; `EvaluateModel` logs `evaluated N
  images in Ts (X img/s)`. Both via `detr::log::Stopwatch`.

## [0.2.2] - 2026-06-10

### Changed
- **Loss uses a paired (O(M)) GIoU instead of pairwise-then-diagonal (O(M²)).**
  `SetCriterion` no longer builds the full `[M,M]` `GeneralizedBoxIou` matrix just
  to take its diagonal; a new `GeneralizedBoxIouPaired` computes GIoU elementwise
  over matched pairs. Numerically identical (proven by `test_box_ops`), so the
  loss and training are unchanged — just less compute/memory. The pairwise
  `GeneralizedBoxIou` stays for the matcher.

## [0.2.1] - 2026-06-10

### Changed
- **Build: dropped the `<torch/torch.h>` umbrella from tensor-only headers**
  (`box_ops` / `target` / `matcher` / `criterion` / `loader` / `pre`+`postprocess`
  / `evaluator` → `<torch/types.h>`; `model.hpp` → `<torch/nn/module.h>` +
  `<torch/types.h>`). ~28% (~2.5s) faster front-end per tensor-only translation
  unit (e.g. `box_ops.cpp` 8.7s→5.6s); no API or behavior change, 79/79 tests pass.

## [0.2.0] - 2026-06-10

Phase 1–2 (in progress). Verified against a real toolchain: portable
cmake/ninja/vcpkg + LibTorch (2.5.1 CPU and 2.7.1+cu128 GPU). The full build
(`-DDETR_ENABLE_TORCH=ON`, 79 tests) passes with zero warnings in project code,
on both CPU and the Blackwell GPU. **28 models registered; 3 validated against
official weights with real, now pycocotools-faithful COCO mAP** (@imgsz800:
detr-r50 41.9, deformable-detr 44.3, conditional-detr 40.7). See
[`docs/STATUS.md`](docs/STATUS.md) for done/remaining.

### Added
- **DETR deep supervision (auxiliary losses).** Every model emits a per-
  intermediate-decoder-layer prediction (`Detections::aux_logits/aux_boxes`), and
  the trainer adds the full set loss — independently Hungarian-matched — on each,
  across all four decoder heads (`detr_head`, `deform_head`, conditional, dab,
  deformable). Training-only (empty at inference, so eval is byte-identical — the
  three validated models reproduce their exact mAP). The largest from-scratch
  training-quality lever. New `test_aux_loss`.
- **FrozenBatchNorm2d backbone** (`models/frozen_batchnorm`) — eval-identical to
  BatchNorm-in-eval (unit-proven), frozen during training to match the DETR
  recipe; also yields a clean weight load (no `num_batches_tracked` mismatches).
- **Training throughput & observability.** Two-group AdamW (backbone lr ×0.1) +
  step `lr_drop`; a parallel batch decoder (was serial single-thread); a
  `detr::log::Stopwatch` primitive; NDJSON records gained timestamp / run-id /
  thread; the train loop logs img/s and the data-vs-compute split.

### Changed
- **COCO eval is now pycocotools-faithful** (`area` field + `iscrowd` ignore).
  Crowd annotations are kept as eval ignore regions (and excluded from training
  targets), and small/medium/large bucketing uses the segmentation `area` field
  instead of the bbox area. Closes the gap to the official metric: detr-r50
  41.5→41.9, deformable-detr 43.8→44.3, conditional-detr 40.3→40.7, with the
  per-size breakdown now matching the published numbers.
- Shared `SinePos` hoisted into `models/pos_embed` (one definition, not five);
  `kPi`→`std::numbers::pi`; hand-rolled `starts_with`→C++20.

### Fixed
- **Weight-load safety.** Predict/export inspect the `LoadReport` and refuse to
  run on 0 matched tensors (a key-mismatched checkpoint previously ran silently
  on random weights); shape-mismatched tensors are logged.
- **Hardened untrusted parsing.** safetensors guards the shape-product and
  byte-size multiplies against integer overflow; the COCO loader rejects
  `file_name` path traversal (absolute / `..`).
- Flaky `test_safetensors` parallel temp-file collision (now pid-qualified).

### Added
- **Deformable-DETR official weights → real COCO mAP.** The HF SenseTime/
  deformable-detr checkpoint loads into our model with **0 unexpected** (a /tmp
  converter renames keys + concatenates the decoder self-attn q/k/v into
  in_proj_weight; the repo is unchanged) and reproduces the published metric:
  **mAP50-95 0.438, mAP50 0.627 on full COCO val** (official 0.445). Correct
  detections on a real image. Second focal-family model validated end to end.

### Changed
- **DRY**: RT-DETR / RF-DETR / DINO now share one deformable detection head
  (`models/deform_head`: query selection + deformable decoder + iterative
  refinement) instead of three copies (~200 lines removed; behaviour unchanged).

### Added
- **DINO + RF-DETR + a ViT backbone.** DINO: a multi-scale deformable encoder +
  IoU-aware query selection + iterative-refinement deformable decoder. RF-DETR:
  the same deformable-decoder head on a new **ViT (DINOv2-style) backbone**
  (`models/vit` — patch embed, 2D sine position, pre-norm blocks). Both
  sigmoid/focal, both GPU-validated (DINO loss 10645→783, RF-DETR 3764→197).
  Contrastive-denoising (DINO) and exact DINOv2 windowing (RF-DETR) are tracked.
- **Conditional-DETR + DAB-DETR.** Conditional-DETR adds the decoupled
  content/spatial decoder cross-attention (Q/K 2× width, V width — a hand-written
  attention). DAB-DETR builds on the same shared decoder layer with 4D anchor-box
  queries, width/height-modulated positional attention, and per-layer iterative
  anchor refinement. Both sigmoid/focal; both train on GPU (loss ↓ ~20×). The
  shared decoder layer lives in `models/cond_decoder`.
- **Size taxonomy (n/s/m/l/x) + RT-DETR size matrix.** ResNet gained BasicBlock
  (R18/R34) alongside Bottleneck, with `feature_channels()`. RT-DETR is registered
  across `rt-detr[v2,v3]-{n,s,m,l,x}` (n=R18@128, s=R18, m=R34, l=R50, x=R101) —
  24 models total. v2/v3 share v1's inference architecture for now (their gains
  are training recipes — tracked). Added `README.md`.
- **RT-DETR (real-time DETR).** The flagship real-time model: a hybrid encoder
  (AIFI transformer on the top level + CCFM CNN cross-scale fusion with RepVGG/
  CSPRepLayer FPN+PAN), IoU-aware query selection from grid anchors, and a
  deformable decoder with 4D reference points + per-layer iterative box refinement
  (sigmoid/focal head). Reuses the shared ResNet + deformable op + focal path.
  Registered as `rt-detr`; forward + focal-train tested, trains on GPU (focal loss
  3534 → 58 / 20 epochs). The deformable op gained a 4D-reference-point path.
- **Multi-scale deformable attention + Deformable-DETR.** `MSDeformAttn` (the
  Deformable-DETR op, via grid_sample) is validated numerically against the torch
  reference (max|Δ| < 1e-5) and is the shared gateway for the deformable family.
  `deformable-detr` is built on it: multi-scale ResNet features (C3–C5 + extra
  level), deformable encoder/decoder, learned queries with reference points, and
  the faithful sigmoid/focal head. Registered (6 models); forward-shape tested.
  Faithful focal-loss training + sigmoid-topk eval + official-weight validation
  are the tracked next steps. The ResNet backbone is now a shared module.
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
