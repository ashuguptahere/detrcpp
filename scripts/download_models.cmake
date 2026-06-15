# download_models.cmake — fetch original-author checkpoints into models/.
#
# Python-free model-zoo downloader. Every entry points at the AUTHORS' OWN release
# from the ORIGINAL upstream repo (never a Hugging Face mirror) — detrcpp loads these
# native `.pth` files directly via each model's UpstreamRemapper (0 missing/0 unexpected).
#
# Usage (run from anywhere):
#   cmake -P scripts/download_models.cmake -- <model-name>      # one model
#   cmake -P scripts/download_models.cmake -- all               # everything
#   cmake -P scripts/download_models.cmake -- list              # print the manifest
#
# Files land in <repo>/models/. Existing files are kept (delete to re-fetch).

cmake_minimum_required(VERSION 3.24)

# ---- Manifest -------------------------------------------------------------------
# One row per registered model: "name|filename|url". Google-Drive assets use the
# direct usercontent form "gdrive:<FILE_ID>" (resolved below) since the authors of
# several repos publish only on Drive.
set(MANIFEST
  # DETR (facebookresearch/detr) — official ResNet-50/101 (+ DC5) COCO checkpoints.
  "detr-r50|detr-r50-e632da11.pth|https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
  "detr-r101|detr-r101-2c7b67e5.pth|https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth"
  "detr-r50-dc5|detr-r50-dc5-f0fb7ef5.pth|https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth"
  "detr-r101-dc5|detr-r101-dc5-a2e86def.pth|https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth"
  # RT-DETR (v1, lyuwenyu/RT-DETR) — native COCO checkpoints, GitHub release assets.
  "rt-detr-s|rtdetr_r18vd_dec3_6x_coco_from_paddle.pth|https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
  "rt-detr-m|rtdetr_r34vd_dec4_6x_coco_from_paddle.pth|https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"
  "rt-detr-l|rtdetr_r50vd_6x_coco_from_paddle.pth|https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"
  "rt-detr-x|rtdetr_r101vd_6x_coco_from_paddle.pth|https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth"
  # RT-DETRv2 (lyuwenyu/RT-DETR) — native EMA checkpoints, GitHub release assets.
  "rt-detrv2-s|rtdetrv2_r18vd_120e_coco_rerun_48.1.pth|https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
  "rt-detrv2-m|rtdetrv2_r34vd_120e_coco_ema.pth|https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth"
  "rt-detrv2-l|rtdetrv2_r50vd_6x_coco_ema.pth|https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth"
  "rt-detrv2-x|rtdetrv2_r101vd_6x_coco_from_paddle.pth|https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth"
  # DEIM (Intellindust-AI-Lab/DEIM) — DEIM-D-FINE (n/s/m/l/x) + DEIM-RT-DETRv2 (s/m/l),
  # the authors' Google-Drive releases. Same graphs as D-FINE / RT-DETRv2 (shared remappers).
  "deim-n|deim_dfine_n.pth|gdrive:1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e"
  "deim-s|deim_dfine_s.pth|gdrive:1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC"
  "deim-m|deim_dfine_m.pth|gdrive:18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8"
  "deim-l|deim_dfine_l.pth|gdrive:1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr"
  "deim-x|deim_dfine_x.pth|gdrive:1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j"
  "deim-rt-s|deim_rt_r18.pth|gdrive:153_JKff6EpFgiLKaqkJsoDcLal_0ux_F"
  "deim-rt-m|deim_rt_r34.pth|gdrive:1O9RjZF6kdFWGv1Etn1Toml4r-YfdMDMM"
  "deim-rt-l|deim_rt_r50.pth|gdrive:1mWknAXD5JYknUQ94WCEvPfXz13jcNOTI"
  # D-FINE (Peterande/D-FINE) — COCO + Objects365->COCO, GitHub release assets.
  "dfine-n|dfine_n_coco.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"
  "dfine-s|dfine_s_coco.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth"
  "dfine-m|dfine_m_coco.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth"
  "dfine-l|dfine_l_coco.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth"
  "dfine-x|dfine_x_coco.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth"
  "dfine-s-obj|dfine_s_obj2coco.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth"
  "dfine-m-obj|dfine_m_obj2coco.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth"
  "dfine-l-obj|dfine_l_obj2coco_e25.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth"
  "dfine-x-obj|dfine_x_obj2coco.pth|https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth"
)

# ---- Arg parsing ----------------------------------------------------------------
# Everything after the literal "--" is our argument; CMake puts script args in
# CMAKE_ARGV{N}. Find the token after "--".
set(REQUEST "")
math(EXPR _last "${CMAKE_ARGC}-1")
foreach(i RANGE 0 ${_last})
  if(DEFINED CMAKE_ARGV${i} AND "${CMAKE_ARGV${i}}" STREQUAL "--")
    math(EXPR _next "${i}+1")
    if(DEFINED CMAKE_ARGV${_next})
      set(REQUEST "${CMAKE_ARGV${_next}}")
    endif()
  endif()
endforeach()
if(REQUEST STREQUAL "")
  message(FATAL_ERROR "usage: cmake -P scripts/download_models.cmake -- <model-name|all|list>")
endif()

get_filename_component(MODELS_DIR "${CMAKE_CURRENT_LIST_DIR}/../models" ABSOLUTE)

# ---- Resolve a possibly-Drive URL to something file(DOWNLOAD) can fetch ----------
function(resolve_url raw out)
  if(raw MATCHES "^gdrive:(.+)$")
    set(${out} "https://drive.usercontent.google.com/download?id=${CMAKE_MATCH_1}&export=download&confirm=t" PARENT_SCOPE)
  else()
    set(${out} "${raw}" PARENT_SCOPE)
  endif()
endfunction()

# ---- One download ---------------------------------------------------------------
function(fetch_one name file url)
  set(dest "${MODELS_DIR}/${file}")
  if(EXISTS "${dest}")
    message(STATUS "${name}: already present (${file})")
    return()
  endif()
  resolve_url("${url}" real)
  message(STATUS "${name}: downloading ${file}")
  # SHOW_PROGRESS trips file(DOWNLOAD) on Google-Drive's usercontent endpoint, so the
  # progress meter is GitHub-only; Drive downloads run quietly.
  if(real MATCHES "drive\\.usercontent\\.google\\.com")
    file(DOWNLOAD "${real}" "${dest}" STATUS st)
  else()
    file(DOWNLOAD "${real}" "${dest}" SHOW_PROGRESS STATUS st)
  endif()
  list(GET st 0 code)
  list(GET st 1 msg)
  if(NOT code EQUAL 0)
    file(REMOVE "${dest}")
    message(FATAL_ERROR "${name}: download failed (${code}: ${msg})")
  endif()
  message(STATUS "${name}: saved -> ${dest}")
endfunction()

# ---- Dispatch -------------------------------------------------------------------
if(REQUEST STREQUAL "list")
  foreach(row ${MANIFEST})
    string(REPLACE "|" ";" parts "${row}")
    list(GET parts 0 name)
    list(GET parts 2 url)
    message("${name}  <-  ${url}")
  endforeach()
  return()
endif()

set(matched FALSE)
foreach(row ${MANIFEST})
  string(REPLACE "|" ";" parts "${row}")
  list(GET parts 0 name)
  list(GET parts 1 file)
  list(GET parts 2 url)
  if(REQUEST STREQUAL "all" OR REQUEST STREQUAL "${name}")
    fetch_one("${name}" "${file}" "${url}")
    set(matched TRUE)
  endif()
endforeach()

if(NOT matched)
  message(FATAL_ERROR "unknown model '${REQUEST}' — run with 'list' to see the manifest")
endif()
