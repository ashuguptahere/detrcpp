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
  file(DOWNLOAD "${real}" "${dest}" SHOW_PROGRESS STATUS st)
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
