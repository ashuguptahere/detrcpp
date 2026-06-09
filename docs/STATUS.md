# detrcpp — Project Status

A concise, honest snapshot of what is **done** and what is **remaining**. The
full task list lives in [`TODO.md`](../TODO.md); the release notes in
[`CHANGELOG.md`](../CHANGELOG.md). Last updated 2026-06-09.

---

## Done

### Foundation
- **C++20** project: CMake + Ninja + vcpkg, Google style (clang-format enforced in
  CI), `tl::expected` errors, RAII, spdlog (NDJSON) logging, CLI11.
- **CLI** verbs fully wired (with LibTorch): `--train`, `--val`, `--test`,
  `--predict`, `--export`, `--list-models`, `--version`. (`--download` is a stub.)
- **CI** green: build matrix (Linux/macOS × debug/release), clang-format check,
  gitleaks, license scan (deny GPL/AGPL/SSPL).
- **GPU (CUDA)**: builds against CUDA LibTorch and runs on NVIDIA Blackwell
  (sm_120) via `--device cuda:0` (~17× faster than CPU). No-root CUDA toolkit
  assembly script (`scripts/setup_cuda_toolkit.sh`).
- **Python-free** repo: weight conversions and tooling stay outside the codebase.

### Weights & export
- **Weight interop**: `.safetensors` reader/writer, `StateDict`/`RawTensor`,
  `WeightRemapper`, LibTorch bridge. `.pth` loader for modern (zip) checkpoints.
- **ONNX export**: hand-written C++ emitter (no `torch.onnx`), gated by an
  onnxruntime numeric **parity** test (`detr`, `detr-r50`, `detr-r101`, ~1e-6).

### Models — 28 registered (10 base architectures + the RT-DETR size matrix)
| family | variants | head |
|--------|----------|------|
| DETR | `detr`, `detr-r50`, `detr-r101`, `detr-r50-dc5`, `detr-r101-dc5` | softmax + no-object |
| Deformable-DETR | `deformable-detr` | sigmoid/focal |
| Conditional-DETR | `conditional-detr` | sigmoid/focal |
| DAB-DETR | `dab-detr` | sigmoid/focal |
| DINO | `dino` | sigmoid/focal |
| RF-DETR | `rf-detr` (ViT backbone) | sigmoid/focal |
| RT-DETR | `rt-detr[v2,v3]-{n,s,m,l,x}` (+ plain = `-l`) | sigmoid/focal |

All forward, train (focal or softmax) on GPU, predict, and evaluate. Shared
building blocks: ResNet (BasicBlock + Bottleneck, dc5), the DETR transformer head,
the decoupled conditional decoder layer, the deformable detection head (query
selection + deformable decoder), multi-scale deformable attention, a ViT backbone.

### Validated against official weights (real COCO mAP, full 5000-image val)
| model | head | ours | official |
|-------|------|------|----------|
| `detr-r50` | softmax | **41.5** | 42.0 |
| `deformable-detr` | sigmoid/focal | **43.8** | 44.5 |
| `conditional-detr` | sigmoid/focal | **40.3** | 40.9 |

Each loads with **0 unexpected** tensors and lands within ~0.7 AP (the gap is
resize-interpolation: our stb bilinear vs PIL). Recipe: convert the HF checkpoint
(rename keys to ours + concatenate split q/k/v into the combined in-proj) and eval
with `--coco91 --aspect`.

### Eval / inference
- COCO mAP with the small/medium/large breakdown (faithful pycocotools reimpl).
- Aspect-preserving eval (`--aspect`), raw COCO-91 ids (`--coco91`).
- Softmax (argmax) and focal (sigmoid top-100) postprocess paths.

---

## Remaining

### Validate the rest of the focal family (needs architecture changes)
- **`dab-detr`**: official uses **PReLU** activation in the FFN + an **encoder
  `query_scale`** modulation. Parameterize the shared decoder layer's activation
  and add the encoder query_scale, then convert HF `IDEA-Research/dab-detr-resnet-50`.
- **`rt-detr`**: official uses a **ResNet-50-VD** backbone (deep 3×3 stem + avg-pool
  downsample). Implement the VD stem, then convert HF `PekingU/rtdetr_r50vd`.

### Training recipes (these models' inference is in; the recipe is not)
- **DN-DETR**: a denoising training recipe on DAB-DETR (noised GT queries + an
  attention mask) — its inference net **is** DAB-DETR.
- **DINO**: contrastive denoising (CDN) training.
- **RT-DETR v2/v3**: discrete sampling / dense-supervision recipes (today v2/v3
  share v1's inference architecture).
- **RF-DETR**: exact DINOv2 backbone fidelity (windowed attention, register tokens)
  + `n/s/m/b/l` sizes.

### Other
- ONNX export for the focal/deformable models (only DETR variants so far).
- Legacy (pre-1.6) `.pth` unpickler in C++ (to avoid the one-off conversion step).
- Full training-loop polish: LR scheduler, resume parity test, AMP.

### Deferred (per the plan)
TensorRT / CoreML / edge export, SAHI sliced inference, the 6 trackers, the
Clay+SDL3 GUI, iOS/Android bindings, the dataset `--download` command, and the
native Parquet dataset format.
