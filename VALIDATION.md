# Validation

Real COCO `val2017` mAP (5000 images) reproduced end-to-end with detrcpp from the
**official published weights**. Each checkpoint loads with `0 shape-mismatched`, and
the measured AP matches the published figure within ~0.4. Run on the GPU build
(`build/gpu`, Release + `libtorch-cu128`) against `--data <coco-root>` on
2026-06-11 (detrcpp 0.13.1).

## Results

| Model | mAP50-95 | mAP50 | mAP75 | Official | Eval flags |
|-------|---------:|------:|------:|---------:|------------|
| `detr-r50`         | **0.419** | 0.623 | 0.444 | ~0.420 | `--coco91 --aspect --imgsz 800` |
| `deformable-detr`  | **0.443** | 0.634 | 0.484 | ~0.445 | `--coco91 --aspect --imgsz 800` |
| `conditional-detr` | **0.407** | 0.616 | 0.431 | ~0.409 | `--coco91 --aspect --imgsz 800` |
| `rt-detr-s`        | **0.463** | 0.635 | 0.502 | ~0.465 | `--imgsz 640` (no `--coco91`, no `--aspect`) |
| `rt-detr-m`        | **0.487** | 0.665 | 0.526 | ~0.489 | `--imgsz 640` (no `--coco91`, no `--aspect`) |
| `rt-detr-l`        | **0.530** | 0.710 | 0.576 | ~0.534 | `--imgsz 640` (no `--coco91`, no `--aspect`) |
| `rt-detr-x`        | **0.542** | 0.725 | 0.585 | ~0.543 | `--imgsz 640` (no `--coco91`, no `--aspect`) |
| `rf-detr-nano`     | **0.482** | 0.672 | 0.516 | 0.484 (AP50 67.6) | `--coco91 --imgsz 384` (no `--aspect`) |

`detr-r50` is loaded from its **legacy (pre-1.6) `.pth`** directly by detrcpp's
pure-C++ unpickler (no Python, no conversion): the load reports `458 tensors,
0 missing / 0 unexpected / 0 shape-mismatched`. The other three load from
safetensors converted from the official Hugging Face checkpoints.

> The remaining ~0.1–0.4 AP gap to the published numbers is the stb-vs-PIL image
> resize interpolation difference (the eval is otherwise pycocotools-faithful:
> `area`-field small/medium/large bucketing + `iscrowd` ignore).

## Weights and how to reproduce them

The official weights are converted to detrcpp's parameter naming by **repo-external**
Python converters under `/tmp` (the repo itself stays Python-free). The `/tmp` paths
are scratch — regenerate them from the Hugging Face source with the listed converter.

| Model | Official source | Converter (`python <script>`) | Converted weights (detrcpp-named) |
|-------|-----------------|-------------------------------|-----------------------------------|
| `detr-r50`         | facebookresearch DETR (legacy `.pth`)        | none — read directly by the C++ legacy unpickler | `/home/origo/detr-tools/detr-r50-official.pth` |
| `deformable-detr`  | `SenseTime/deformable-detr` `model.safetensors` | `/tmp/convert_defdetr.py` (in: `/tmp/hf_defdetr.safetensors`) | `/tmp/defdetr_official.safetensors` |
| `conditional-detr` | `microsoft/conditional-detr-resnet-50`       | `/tmp/convert_cond.py`    | `/tmp/cond_official.safetensors` |
| `rt-detr-s`        | `PekingU/rtdetr_r18vd`                        | `/tmp/convert_rtdetr_size.py <in> <out>` | `/tmp/rtdetr_r18vd.safetensors` (509 tensors) |
| `rt-detr-m`        | `PekingU/rtdetr_r34vd`                        | `/tmp/convert_rtdetr_size.py <in> <out>` | `/tmp/rtdetr_r34vd.safetensors` (619 tensors) |
| `rt-detr-l`        | `PekingU/rtdetr_r50vd`                        | `/tmp/convert_rtdetr.py` (in: `/tmp/hf_rtdetr.safetensors`) | `/tmp/rtdetr_official.safetensors` |
| `rt-detr-x`        | `PekingU/rtdetr_r101vd`                       | `/tmp/convert_rtdetr_size.py <in> <out>` | `/tmp/rtdetr_r101vd.safetensors` (990 tensors) |
| `rf-detr-nano`     | `stevenbucaille/rf-detr-nano` (HF mirror of the Roboflow `.pth`) | `/tmp/convert_rfdetr_full.py` | `/tmp/rfdetr_full.safetensors` (328 tensors) |

The `rt-detr-{s,m,x}` checkpoints share one I/O-parameterized converter
(`convert_rtdetr_size.py`); it is backbone-agnostic (BasicBlock R18/R34 vs Bottleneck
R101) and counts the decoder depth (3/4/6) from the source.

A converter downloads (or reads) the HF safetensors, renames keys to detrcpp's
module tree (and concatenates split `q/k/v` projections into the
`nn::MultiheadAttention` `in_proj`), and writes a detrcpp-named safetensors that
loads with an identity remapper.

## Command

```sh
detrcpp --val -m <model> -w <weights> --data <coco-root> <flags> --device cuda:0
```

The DETR family uses `--coco91 --aspect --imgsz 800` (91-class COCO ids,
aspect-preserving 800-short-side resize). **RT-DETR differs**: contiguous 80
classes (no `--coco91`), square raw-`[0,1]` resize (no `--aspect`,
`imagenet_norm=false`), `--imgsz 640`. **RF-DETR differs again**: 91-class COCO ids
(`--coco91`), ImageNet-normalized **square** resize at the native `--imgsz 384`
(no `--aspect`) — the faithful `rf-detr-nano` model (DINOv2-windowed backbone + C2f
projector + two-stage deformable decoder), not the placeholder-ViT `rf-detr-{n..x}`.

## Registered-but-not-yet-validated variants

- **`rt-detr-{s,m,l,x}`** are all **validated** above (R18/R34/R50/R101-vd). `rt-detr-n`
  is detrcpp's own nano (R18 @ width 128) with no published weights. The
  **`rt-detrv2-*` / `rt-detrv3-*`** matrices share v1's inference graph (their gains
  are training recipes — discrete sampling / dense supervision), so v1's validated
  weights load into them; PekingU's separate v2 checkpoints are a tracked follow-up.
- **`rf-detr-nano`** (the faithful model) is **validated** above. The other faithful
  sizes (small/medium/large/xlarge) share its code but need their own published
  Roboflow weights to validate, so they are not registered yet. The placeholder-ViT
  `rf-detr` / `rf-detr-cdn` / `rf-detr-{n,s,m,l,x}` entries train but do **not** load
  official weights (no DINOv2 windowing / C2f projector / two-stage decoder); they
  remain for the training recipes (e.g. `rf-detr-cdn` contrastive denoising).
