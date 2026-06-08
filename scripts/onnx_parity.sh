#!/bin/sh
# Copyright 2026 detrcpp authors. Apache-2.0.
#
# End-to-end ONNX parity gate. Because the ONNX exporter is torch-free and the
# main model is LibTorch, the check spans two build configurations:
#   1. torch build  -> detr-golden: dump weights + a fixed input + reference out
#   2. onnx  build  -> detr-parity: export the same weights to ONNX, run them in
#                       onnxruntime, and compare to the reference.
# Exits non-zero if max|Δ| exceeds the tolerance.
#
# Env (with the defaults used during development):
#   VCPKG_ROOT, LIBTORCH_ROOT, ONNXRUNTIME_ROOT, CMAKE/ninja on PATH.
set -e

CFG="${1:-configs/models/detr-tiny.yaml}"
DIR="${2:-/tmp/detr_parity}"
TOL="${3:-1e-3}"

: "${VCPKG_ROOT:?set VCPKG_ROOT}"
: "${LIBTORCH_ROOT:?set LIBTORCH_ROOT (path to libtorch)}"
: "${ONNXRUNTIME_ROOT:?set ONNXRUNTIME_ROOT (path to onnxruntime)}"

JOBS=$(( $(nproc) - 2 )); [ "$JOBS" -lt 1 ] && JOBS=1
TC="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

echo "== [1/4] build torch detr-golden =="
cmake -S . -B build/torch -G Ninja -DCMAKE_TOOLCHAIN_FILE="$TC" \
  -DVCPKG_MANIFEST_FEATURES=tests -DCMAKE_BUILD_TYPE=Debug \
  -DDETR_ENABLE_TORCH=ON -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT" >/dev/null
cmake --build build/torch --target detr-golden -j "$JOBS"

echo "== [2/4] generate golden =="
LD_LIBRARY_PATH="$LIBTORCH_ROOT/lib:$LD_LIBRARY_PATH" \
  build/torch/apps/detr-golden/detr-golden "$CFG" "$DIR"

echo "== [3/4] build onnx detr-parity =="
cmake -S . -B build/onnx -G Ninja -DCMAKE_TOOLCHAIN_FILE="$TC" \
  -DVCPKG_MANIFEST_FEATURES="tests;onnx" -DCMAKE_BUILD_TYPE=Debug \
  -DDETR_ENABLE_ONNX=ON -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT" >/dev/null
cmake --build build/onnx --target detr-parity -j "$JOBS"

echo "== [4/4] parity: ONNX vs LibTorch =="
LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH" \
  build/onnx/apps/detr-parity/detr-parity "$CFG" "$DIR" "$TOL"
