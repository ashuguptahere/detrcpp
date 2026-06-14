# license_scan.cmake — fails the configure step if any dependency carries a
# license that is not on the permissive allowlist (deny-by-default / fail-closed).
#
# Design notes (these matter — an earlier version was a silent no-op):
#   * Dependencies are vendored from source by FetchContent into build/_deps/<name>-src.
#     We pin exact tags, so each dep's SPDX license is asserted in DETR_DEP_LICENSES
#     below (auditable, in-tree) and cross-checked against the actual LICENSE/COPYING
#     file shipped in the fetched source — a missing license file is itself a failure.
#   * The check is ALLOWLIST-based: a license token passes only if it is on the
#     allowlist; anything unknown is a violation. The denylist is an explicit
#     fast-fail with a clearer message.
#   * If the scanner inspected ZERO dependencies while _deps sources exist, that is a
#     FATAL error — a compliance gate that inspected nothing must never report
#     "passed" (false confidence is worse than no gate).
#   * Defense in depth: the fetched LICENSE text is substring-scanned for denylisted
#     license names, catching a dep that silently relicenses between pin bumps.

# Asserted SPDX license per vendored dependency (key = FetchContent name = the
# build/_deps/<key>-src directory). Keep in sync with cmake/dependencies.cmake.
set(DETR_DEP_LICENSES
    "fmt=MIT"
    "spdlog=MIT"
    "cli11=BSD-3-Clause"
    "expected=CC0-1.0"
    "simdjson=Apache-2.0"
    "yaml-cpp=MIT"
    "googletest=BSD-3-Clause"
    "benchmark=Apache-2.0"
    "protobuf=BSD-3-Clause"
    "onnx=Apache-2.0"
    CACHE STRING "SPDX license asserted for each FetchContent dependency")

set(DETR_LICENSE_ALLOWLIST
    "Apache-2.0"
    "MIT"
    "MIT-0"
    "BSD-2-Clause"
    "BSD-3-Clause"
    "BSD-3-Clause-Clear"
    "Zlib"
    "libpng-2.0"
    "ISC"
    "BSL-1.0"          # Boost
    "MPL-2.0"          # weak/file-scoped copyleft — acceptable for linking
    "Unlicense"
    "CC0-1.0"
    "0BSD"
    "NCSA"
    "Apache-2.0-WITH-LLVM-exception"
    CACHE STRING "SPDX license tokens permitted in the dependency graph")

set(DETR_LICENSE_DENYLIST
    "GPL-2.0-only" "GPL-2.0-or-later"
    "GPL-3.0-only" "GPL-3.0-or-later"
    "LGPL-2.1-only" "LGPL-2.1-or-later"   # LGPL static-link grey-area: deny by default
    "LGPL-3.0-only" "LGPL-3.0-or-later"
    "AGPL-3.0-only" "AGPL-3.0-or-later"
    "SSPL-1.0"
    "BUSL-1.1"
    "Commons-Clause"
    "CC-BY-NC-4.0"
    CACHE STRING "SPDX license tokens explicitly forbidden (fast-fail)")

# Whether an unrecognized license (not on allow- or deny-list) is fatal.
option(DETR_LICENSE_STRICT "Treat unknown licenses as violations (fail-closed)" ON)

# Splits an SPDX license expression into bare tokens, dropping the AND/OR/WITH
# operators, parentheses, and SPDX placeholders. Result is set in ${out_var}.
function(_detr_spdx_tokens expr out_var)
  string(REGEX REPLACE "[()]" " " expr "${expr}")
  string(REGEX REPLACE "([ \t]+(AND|OR|WITH)[ \t]+)" ";" expr "${expr}")
  string(REPLACE " " "" expr "${expr}")
  set(_tokens "")
  foreach(_t IN LISTS expr)
    string(STRIP "${_t}" _t)
    if(_t STREQUAL "" OR _t STREQUAL "NOASSERTION" OR _t STREQUAL "NONE")
      continue()
    endif()
    list(APPEND _tokens "${_t}")
  endforeach()
  set(${out_var} "${_tokens}" PARENT_SCOPE)
endfunction()

# Classifies a single SPDX token. Appends a human-readable string to the
# parent-scope list named by ${violations_var} when the token is not permitted.
function(_detr_classify_token token source violations_var)
  if(token IN_LIST DETR_LICENSE_DENYLIST)
    set(_v "${${violations_var}}")
    list(APPEND _v "${source}: DENIED license '${token}'")
    set(${violations_var} "${_v}" PARENT_SCOPE)
  elseif(NOT token IN_LIST DETR_LICENSE_ALLOWLIST)
    if(DETR_LICENSE_STRICT)
      set(_v "${${violations_var}}")
      list(APPEND _v "${source}: UNKNOWN license '${token}' (not on allowlist)")
      set(${violations_var} "${_v}" PARENT_SCOPE)
    else()
      message(WARNING "detrcpp: ${source}: unknown license '${token}' (allowed: STRICT off)")
    endif()
  endif()
endfunction()

function(detr_license_scan)
  set(_base "${FETCHCONTENT_BASE_DIR}")
  if(NOT _base)
    set(_base "${CMAKE_BINARY_DIR}/_deps")
  endif()

  file(GLOB _srcdirs LIST_DIRECTORIES true "${_base}/*-src")
  if(NOT _srcdirs)
    message(STATUS "detrcpp: license scan skipped (no fetched deps yet — first configure).")
    return()
  endif()

  set(_violations "")
  set(_checked 0)

  foreach(_dir IN LISTS _srcdirs)
    if(NOT IS_DIRECTORY "${_dir}")
      continue()
    endif()
    get_filename_component(_leaf "${_dir}" NAME)
    string(REGEX REPLACE "-src$" "" _name "${_leaf}")

    # Asserted SPDX for this dependency (must be declared — unknown deps fail closed).
    set(_spdx "")
    foreach(_kv IN LISTS DETR_DEP_LICENSES)
      if(_kv MATCHES "^${_name}=(.+)$")
        set(_spdx "${CMAKE_MATCH_1}")
      endif()
    endforeach()
    if(_spdx STREQUAL "")
      list(APPEND _violations
        "${_name}: fetched dependency has no asserted SPDX in DETR_DEP_LICENSES")
      continue()
    endif()

    # The gate must inspect a real license file shipped in the source.
    file(GLOB _lic_files "${_dir}/LICENSE*" "${_dir}/COPYING*" "${_dir}/LICENCE*")
    if(NOT _lic_files)
      list(APPEND _violations
        "${_name}: asserted ${_spdx} but no LICENSE/COPYING file in ${_dir}")
      continue()
    endif()

    # Classify the asserted SPDX expression against the allow/deny lists.
    _detr_spdx_tokens("${_spdx}" _tokens)
    foreach(_tok IN LISTS _tokens)
      _detr_classify_token("${_tok}" "${_name}" _violations)
    endforeach()

    # Defense in depth: substring-scan the actual license text for denylisted names.
    foreach(_lf IN LISTS _lic_files)
      file(READ "${_lf}" _text)
      foreach(_bad IN LISTS DETR_LICENSE_DENYLIST)
        string(REPLACE "+" "" _needle "${_bad}")
        string(REGEX REPLACE "-(only|or-later)$" "" _needle "${_needle}")
        if(_text MATCHES "${_needle}")
          list(APPEND _violations "${_name}: LICENSE text mentions denylisted '${_bad}'")
        endif()
      endforeach()
    endforeach()

    math(EXPR _checked "${_checked} + 1")
  endforeach()

  if(_checked EQUAL 0 AND NOT _violations)
    message(FATAL_ERROR
      "detrcpp: license scan inspected NO dependencies under ${_base} — refusing to "
      "report success.")
  endif()

  list(REMOVE_DUPLICATES _violations)
  if(_violations)
    string(REPLACE ";" "\n  " _pretty "${_violations}")
    message(FATAL_ERROR
      "detrcpp: license scan FAILED — disallowed/unknown licenses in dep graph:\n"
      "  ${_pretty}\n"
      "Fix the pin in cmake/dependencies.cmake, update DETR_DEP_LICENSES, or adjust "
      "DETR_LICENSE_ALLOWLIST / DETR_LICENSE_DENYLIST.")
  endif()

  message(STATUS "detrcpp: license scan passed (${_checked} dependencies checked).")
endfunction()
