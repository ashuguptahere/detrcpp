#!/bin/sh
# Copyright 2026 detrcpp authors. Apache-2.0.
#
# Assemble a CUDA 12.8 build toolkit WITHOUT root, for building against the CUDA
# LibTorch (cu128) on machines where you can't apt-install the toolkit. Caffe2's
# CMake needs a real `nvcc` plus the static cudart libs and the NVTX lib, which
# the pip `nvidia-*-cu12` wheels alone don't fully provide — so we also pull a
# few NVIDIA redistrib archives. The result is a standard <root>/{bin,include,
# lib64} layout that `find_package(CUDAToolkit)` accepts.
#
# Runtime needs only the cu128 LibTorch (it bundles its own CUDA libs) + the
# driver; this toolkit is a BUILD-time dependency.
#
# Usage:  scripts/setup_cuda_toolkit.sh [ROOT]   (default: $DETR_TOOLS/cuda)
set -e

ROOT="${1:-${DETR_TOOLS:-$HOME/detr-tools}/cuda}"
VENV="${CUDA_WHEEL_VENV:-/tmp/cuda-wheels-venv}"
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
TMP="$(mktemp -d)"
echo "== assembling CUDA 12.8 toolkit at $ROOT =="

# 1) Component libraries + headers via pip wheels (no root).
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet \
  nvidia-cuda-runtime-cu12==12.8.57 nvidia-cuda-cccl-cu12==12.8.55 \
  nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-nvrtc-cu12==12.8.61 \
  nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
NV="$VENV/lib/python3.12/site-packages/nvidia"

# 2) nvcc, static cudart (+ libcuda stub), and NVTX from the redistrib archives.
for comp in "cuda_nvcc/cuda_nvcc-linux-x86_64-12.8.93" \
            "cuda_cudart/cuda_cudart-linux-x86_64-12.8.90" \
            "cuda_nvtx/cuda_nvtx-linux-x86_64-12.8.90"; do
  name="${comp#*/}"
  curl -sSL "$REDIST/${comp%/*}/linux-x86_64/${name}-archive.tar.xz" | tar xJ -C "$TMP"
done

# 3) Lay it out as a single toolkit root.
mkdir -p "$ROOT/bin" "$ROOT/include" "$ROOT/lib64/stubs" "$ROOT/nvvm"
NVCC="$TMP/cuda_nvcc-linux-x86_64-12.8.93-archive"
CUDART="$TMP/cuda_cudart-linux-x86_64-12.8.90-archive"
NVTX="$TMP/cuda_nvtx-linux-x86_64-12.8.90-archive"
cp -r "$NVCC/bin/." "$ROOT/bin/"
cp -r "$NVCC/nvvm/." "$ROOT/nvvm/"
for d in "$NV"/*/include "$NVCC/include" "$CUDART/include" "$NVTX/include"; do
  cp -rn "$d/." "$ROOT/include/" 2>/dev/null || true
done
for d in "$NV"/*/lib "$NVTX/lib"; do cp -rn "$d/." "$ROOT/lib64/" 2>/dev/null || true; done
cp "$CUDART/lib/libcudart_static.a" "$CUDART/lib/libcudadevrt.a" "$ROOT/lib64/"
cp -r "$CUDART/lib/stubs/." "$ROOT/lib64/stubs/"
printf '{"cuda":{"version":"12.8.0"},"cuda_cudart":{"version":"12.8.57"}}\n' > "$ROOT/version.json"
# Unversioned .so symlinks for the linker.
for f in libcudart libcublas libcublasLt libcufft libcurand libcusparse libcusolver libnvrtc libnvToolsExt; do
  real=$(ls "$ROOT/lib64/${f}.so."* 2>/dev/null | head -1)
  [ -n "$real" ] && ln -sf "$(basename "$real")" "$ROOT/lib64/${f}.so"
done
rm -rf "$TMP"
echo "== done: configure with -DCUDAToolkit_ROOT=$ROOT =="
