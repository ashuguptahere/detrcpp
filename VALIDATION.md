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
| `rt-detr-l`        | **0.530** | 0.710 | 0.576 | ~0.534 | `--imgsz 640` (no `--coco91`, no `--aspect`) |

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
| `rt-detr-l`        | `PekingU/rtdetr_r50vd`                        | `/tmp/convert_rtdetr.py` (in: `/tmp/hf_rtdetr.safetensors`) | `/tmp/rtdetr_official.safetensors` |

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
`imagenet_norm=false`), `--imgsz 640`.

## Registered-but-not-yet-validated variants

- **`rt-detr-{n,s,m,x}`** (backbones r18 / r18 / r34 / r101-vd) and the
  **`rt-detrv2-*` / `rt-detrv3-*`** matrices are all registered, but only
  `rt-detr-l` (r50vd) has a downloaded official checkpoint here. PekingU publishes
  `rtdetr_r18vd` / `r34vd` / `r101vd` — download each, run `convert_rtdetr.py`, and
  validate with the RT-DETR flags above. (v2/v3 share v1's inference graph today;
  their gains are training recipes — discrete sampling / dense supervision.)
- **`rf-detr` / `rf-detr-cdn`** are registered (a single ViT config) but **not**
  validated against official Roboflow weights, and the per-size variants
  (nano/small/medium/large) are **not** separate registry entries — they would be
  config overrides of `vit_embed` / `vit_depth` / `vit_heads`. Validating RF-DETR
  and registering its size matrix is open work.
