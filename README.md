# detrcpp

A modern, permissively-licensed **C++20 framework for DETR-family object
detection** — train, evaluate, predict, and export, across the whole DETR lineage,
from one clean CLI. No Python in the codebase, no Ultralytics, LibTorch under the
hood.

> Status: active development (Phase 2). The pieces below are real and verified
> against a live toolchain — see **Verified results**. Some variants are
> architecture-complete with training-recipe/official-weight work still tracked in
> [`TODO.md`](TODO.md); the README is honest about what is and isn't done.

---

## Highlights

- **Four models reproduce official COCO mAP from real weights** (full 5000-image
  val, within ~0.4 AP of published): `detr-r50` **0.419** (0.420), `deformable-detr`
  **0.443** (0.445), `conditional-detr` **0.407** (0.409), `rt-detr-l` **0.530**
  (0.534) — across the softmax and sigmoid/focal heads. `detr-r50` loads its
  **legacy pre-1.6 `.pth`** through a pure-C++ unpickler (no Python). See
  [VALIDATION.md](VALIDATION.md) for the per-model weights, converters, and recipes.
- **Python-free ONNX export with numeric parity** — a hand-written C++ ONNX
  emitter matches LibTorch to `max|Δ| ~1e-6` (verified in onnxruntime).
- **GPU (CUDA) on NVIDIA Blackwell** — `--device cuda:0`, ~17× faster than CPU.
- **The deformable family** — multi-scale deformable attention (validated vs the
  PyTorch reference to `<1e-5`), powering Deformable-DETR, RT-DETR, DINO, RF-DETR.
- **One registry, many models** — adding a variant is one file + one registration.

See [`docs/STATUS.md`](docs/STATUS.md) for a full done/remaining breakdown.

## Models

28 registered variants today (10 base architectures + the RT-DETR size matrix).
The size suffix is the convention going forward:

| size | backbone        |
|------|-----------------|
| `n`  | ResNet-18 (width 128, nano) |
| `s`  | ResNet-18       |
| `m`  | ResNet-34       |
| `l`  | ResNet-50       |
| `x`  | ResNet-101      |

| family | variants | classifier | status |
|--------|----------|-----------|--------|
| **DETR** | `detr` (compact), `detr-r50`, `detr-r101`, `detr-r50-dc5`, `detr-r101-dc5` | softmax + no-object | trains, predicts, evals; **official weights load + real mAP**; ONNX parity |
| **Deformable-DETR** | `deformable-detr` | sigmoid/focal | **official weights load (0 missing) → real COCO mAP 0.443** (official 0.445); GPU-validated |
| **Conditional-DETR** | `conditional-detr` | sigmoid/focal | **official weights load → real COCO mAP 0.407** (official 0.409); GPU-validated |
| **DAB-DETR** | `dab-detr` | sigmoid/focal | trains (GPU-validated); 4D anchor queries + HW-modulated attn + iterative refinement |
| **DINO** | `dino` | sigmoid/focal | trains (GPU-validated); deformable encoder + query selection + iterative decoder (CDN training tracked) |
| **RF-DETR** | `rf-detr` | sigmoid/focal | trains (GPU-validated); **ViT backbone** + multi-scale projection + deformable decoder |
| **RT-DETR** | `rt-detr[v2,v3]-{n,s,m,l,x}` (+ plain `rt-detr[v2,v3]` = `-l`) | sigmoid/focal | **`rt-detr-l` official weights load (0 missing) → real COCO mAP 0.530** (official 0.534); GPU-validated. v2 adds discrete sampling, v3 adds dense supervision (training recipes) |

**DN-DETR** is DAB-DETR + a *denoising training* recipe (noised GT queries + an
attention mask) — its inference net == DAB-DETR, so it's a training mode, not a
separate model (tracked).

Roadmap (in [`TODO.md`](TODO.md)): DN-DETR denoising + DINO CDN training; RT-DETR
v2/v3 recipes; ViT/DINOv2 fidelity for RF-DETR; per-variant official-weight validation.

```sh
detrcpp --list-models          # the full table with imgsz / queries / classes / license
```

## Quick start

### Build (CPU)

Uses CMake + Ninja + LibTorch. No package manager — every other dependency is
vendored from source by CMake **FetchContent** (pinned tags; see
[`cmake/dependencies.cmake`](cmake/dependencies.cmake)) on first configure.

```sh
cmake --preset debug -DDETR_ENABLE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build/debug -j"$(( $(nproc) - 2 ))"
ctest --test-dir build/debug --output-on-failure
```

The lightweight build (omit `-DDETR_ENABLE_TORCH`) drops LibTorch and the models —
useful for the data/weights/eval libraries and their tests.

### Build (GPU / CUDA)

Needs a CUDA LibTorch and a CUDA toolkit. `scripts/setup_cuda_toolkit.sh`
assembles a CUDA 12.8 build toolkit with no root (for Blackwell, sm_120). Then
configure with `-DCMAKE_PREFIX_PATH=<cuda-libtorch> -DCUDAToolkit_ROOT=<toolkit>`
and run with `--device cuda:0`.

## CLI

```sh
detrcpp --train   -m rt-detr-l -s 42 --data path/to/coco -e 12 [-d cuda:0]
detrcpp --val     -m detr-r50  -w weights.safetensors --data coco --coco91 --aspect
detrcpp --predict -m detr-r50  -w weights.safetensors -i image.jpg --conf 0.7
detrcpp --export  onnx -m detr -w weights.safetensors      # via detrcpp-export
detrcpp --list-models | --version | --help
```

- `-w` accepts `.safetensors` or modern `.pth` (auto-detected). Legacy (pre-1.6)
  `.pth` is detected and reported (no C++ loader exists yet).
- `--coco91` evaluates with raw COCO-91 class ids (for official DETR weights);
  `--aspect` uses DETR's aspect-preserving eval resize.
- Eval reports `mAP50-95`, `mAP50`, `mAP75`, and the small/medium/large breakdown.

## Verified results

| what | result |
|------|--------|
| `detr-r50` official weights, full COCO val (5000) | mAP50-95 **0.415**, mAP50 **0.617** (official 0.420 / 0.624) |
| `deformable-detr` official weights, full COCO val (5000) | mAP50-95 **0.438**, mAP50 **0.627** (official 0.445) |
| `conditional-detr` official weights, full COCO val (5000) | mAP50-95 **0.403**, mAP50 **0.610** (official 0.409) |
| ONNX export vs LibTorch (`detr`, `detr-r50`, `detr-r101`) | `max\|Δ\|` 5e-7 – 1e-6 |
| deformable attention core vs PyTorch reference | `max\|Δ\|` < 1e-5 |
| Deformable-DETR / RT-DETR training (GPU, focal) | loss ↓ ~60× over a short overfit |
| GPU speedup (500-image eval) | ~17× vs CPU |

## Design

- **C++20**, Google style (clang-format enforced in CI), `tl::expected` errors,
  RAII, spdlog (NDJSON) logging, CLI11.
- **Registry** (`models::IModel` + `RegisterBuiltins`) is the single extension
  point — a new model is one file + one registration (SOLID open/closed).
- **Shared building blocks**: a torchvision-compatible ResNet (BasicBlock +
  Bottleneck), the DETR transformer head, multi-scale deformable attention, and a
  softmax-or-sigmoid/focal classification path selected per model.
- **Weights**: `.safetensors` as the Python-free interchange, with an
  `UpstreamRemapper` per model to load original-repo checkpoints 1:1.
- **ONNX**: a hand-written C++ emitter (no `torch.onnx`, no Python) gated by a
  numeric onnxruntime parity test.

## License

Apache-2.0. Dependencies are restricted to Apache-2.0-compatible permissive
licenses (CI fails on GPL/AGPL/SSPL). See [`CHANGELOG.md`](CHANGELOG.md) for the
full set.
