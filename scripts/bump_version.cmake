# Copyright 2026 detrcpp authors. Apache-2.0.
#
# Bumps the centralized project version (MAJOR.MINOR.PATCH). Pure CMake — no
# Python, no shell. Run it with `cmake -P`:
#
#   cmake -P scripts/bump_version.cmake -- patch       # 0.1.0 -> 0.1.1
#   cmake -P scripts/bump_version.cmake -- minor       # 0.1.1 -> 0.2.0
#   cmake -P scripts/bump_version.cmake -- major       # 0.2.0 -> 1.0.0
#   cmake -P scripts/bump_version.cmake -- --set 2.3.4
#   cmake -P scripts/bump_version.cmake -- --show
#
# The top-level VERSION file is the single source of truth. This script also
# promotes the CHANGELOG.md [Unreleased] section to the new version.

cmake_minimum_required(VERSION 3.24)

# --- locate repo root (this script lives in <root>/scripts/) ---
get_filename_component(_script_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_root "${_script_dir}" DIRECTORY)
set(_version_file "${_root}/VERSION")
set(_changelog    "${_root}/CHANGELOG.md")

# --- collect args after the script path (ARGV0=cmake, 1=-P, 2=script) ---
# A leading "--" is accepted but optional, so both of these work:
#   cmake -P scripts/bump_version.cmake -- patch
#   cmake -P scripts/bump_version.cmake patch
set(_args "")
if(CMAKE_ARGC GREATER 3)
  math(EXPR _last "${CMAKE_ARGC} - 1")
  foreach(_i RANGE 3 ${_last})
    list(APPEND _args "${CMAKE_ARGV${_i}}")
  endforeach()
endif()
if(_args)
  list(GET _args 0 _first)
  if(_first STREQUAL "--")
    list(REMOVE_AT _args 0)
  endif()
endif()

if(_args STREQUAL "")
  message(FATAL_ERROR "usage: cmake -P scripts/bump_version.cmake -- {major|minor|patch|--set X.Y.Z|--show}")
endif()

# --- read + validate current version ---
if(NOT EXISTS "${_version_file}")
  message(FATAL_ERROR "VERSION file not found at ${_version_file}")
endif()
file(STRINGS "${_version_file}" _raw LIMIT_COUNT 1)
string(STRIP "${_raw}" _cur)
if(NOT _cur MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
  message(FATAL_ERROR "VERSION is not MAJOR.MINOR.PATCH: '${_cur}'")
endif()
set(_major "${CMAKE_MATCH_1}")
set(_minor "${CMAKE_MATCH_2}")
set(_patch "${CMAKE_MATCH_3}")

list(GET _args 0 _cmd)

if(_cmd STREQUAL "--show")
  message(STATUS "${_major}.${_minor}.${_patch}")
  return()
elseif(_cmd STREQUAL "--set")
  list(LENGTH _args _n)
  if(_n LESS 2)
    message(FATAL_ERROR "--set requires X.Y.Z")
  endif()
  list(GET _args 1 _explicit)
  if(NOT _explicit MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
    message(FATAL_ERROR "--set expects MAJOR.MINOR.PATCH, got '${_explicit}'")
  endif()
  set(_major "${CMAKE_MATCH_1}")
  set(_minor "${CMAKE_MATCH_2}")
  set(_patch "${CMAKE_MATCH_3}")
elseif(_cmd STREQUAL "major")
  math(EXPR _major "${_major} + 1")
  set(_minor 0)
  set(_patch 0)
elseif(_cmd STREQUAL "minor")
  math(EXPR _minor "${_minor} + 1")
  set(_patch 0)
elseif(_cmd STREQUAL "patch")
  math(EXPR _patch "${_patch} + 1")
else()
  message(FATAL_ERROR "unknown command '${_cmd}' (use major|minor|patch|--set|--show)")
endif()

set(_new "${_major}.${_minor}.${_patch}")

# --- write VERSION ---
file(WRITE "${_version_file}" "${_new}\n")

# --- promote CHANGELOG [Unreleased] -> new version ---
if(EXISTS "${_changelog}")
  file(READ "${_changelog}" _cl)
  if(_cl MATCHES "## \\[Unreleased\\]\n")
    string(TIMESTAMP _today "%Y-%m-%d")
    set(_fresh "## [Unreleased]\n\n### Added\n\n### Changed\n\n### Fixed\n\n## [${_new}] - ${_today}\n")
    string(REPLACE "## [Unreleased]\n" "${_fresh}" _cl "${_cl}")
    file(WRITE "${_changelog}" "${_cl}")
  endif()
endif()

message(STATUS "Bumped ${_cur} -> ${_new}")
message(STATUS "Re-run CMake configure to regenerate detr/version.hpp.")
