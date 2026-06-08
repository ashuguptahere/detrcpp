# license_scan.cmake — fails the configure step if any dependency carries a
# license that is not on the permissive allowlist (deny-by-default / fail-closed).
#
# Design notes (these matter — an earlier version was a silent no-op):
#   * Manifest-mode vcpkg does NOT write classic `CONTROL` files into
#     vcpkg_installed. It writes an SPDX SBOM at
#     `vcpkg_installed/<triplet>/share/<port>/vcpkg.spdx.json` and a free-text
#     `.../share/<port>/copyright` file. We read the SBOM's SPDX license
#     expression (structured, reliable) and fall back to substring-scanning the
#     copyright text for denylisted names.
#   * The check is ALLOWLIST-based: a license token passes only if it is on the
#     allowlist; anything unknown is a violation. The denylist is an explicit
#     fast-fail with a clearer message.
#   * If the scanner finds ZERO metadata while vcpkg_installed exists, that is a
#     FATAL error — a compliance gate that inspected nothing must never report
#     "passed" (false confidence is worse than no gate).

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
  if(NOT DEFINED ENV{VCPKG_ROOT})
    message(STATUS "detrcpp: license scan skipped (VCPKG_ROOT not set).")
    return()
  endif()

  set(_installed "${CMAKE_BINARY_DIR}/vcpkg_installed")
  if(NOT EXISTS "${_installed}")
    message(STATUS "detrcpp: license scan skipped (no vcpkg_installed yet — first configure).")
    return()
  endif()

  file(GLOB_RECURSE _sbom_files  "${_installed}/*/share/*/vcpkg.spdx.json")
  file(GLOB_RECURSE _copyrights  "${_installed}/*/share/*/copyright")

  list(LENGTH _sbom_files _n_sbom)
  list(LENGTH _copyrights _n_copy)

  # A gate that inspected nothing must not pass silently.
  if(_n_sbom EQUAL 0 AND _n_copy EQUAL 0)
    message(FATAL_ERROR
      "detrcpp: license scan found NO dependency metadata under ${_installed} "
      "(expected share/*/vcpkg.spdx.json or share/*/copyright). The compliance "
      "gate cannot run — refusing to report success. Check the vcpkg layout.")
  endif()

  set(_violations "")

  # Preferred path: structured SPDX from the SBOM.
  foreach(_sbom IN LISTS _sbom_files)
    file(READ "${_sbom}" _json)
    string(JSON _npkg ERROR_VARIABLE _e LENGTH "${_json}" packages)
    if(_e OR _npkg STREQUAL "" OR _npkg LESS_EQUAL 0)
      continue()
    endif()
    math(EXPR _last "${_npkg} - 1")
    foreach(_i RANGE 0 ${_last})
      string(JSON _pkg ERROR_VARIABLE _e GET "${_json}" packages ${_i})
      if(_e)
        continue()
      endif()
      string(JSON _lic ERROR_VARIABLE _e GET "${_pkg}" licenseConcluded)
      if(_e OR _lic STREQUAL "")
        string(JSON _lic ERROR_VARIABLE _e2 GET "${_pkg}" licenseDeclared)
      endif()
      if(_lic STREQUAL "" OR _lic STREQUAL "NOASSERTION")
        continue()
      endif()
      _detr_spdx_tokens("${_lic}" _tokens)
      foreach(_tok IN LISTS _tokens)
        _detr_classify_token("${_tok}" "${_sbom}" _violations)
      endforeach()
    endforeach()
  endforeach()

  # Defense in depth: substring-scan free-text copyright files for denylisted
  # names (catches GPL/AGPL even when no SBOM is present).
  foreach(_copy IN LISTS _copyrights)
    file(READ "${_copy}" _text)
    foreach(_bad IN LISTS DETR_LICENSE_DENYLIST)
      string(REPLACE "+" "" _needle "${_bad}")
      string(REGEX REPLACE "-(only|or-later)$" "" _needle "${_needle}")
      if(_text MATCHES "${_needle}")
        list(APPEND _violations "${_copy}: copyright text mentions denylisted '${_bad}'")
      endif()
    endforeach()
  endforeach()

  list(REMOVE_DUPLICATES _violations)
  if(_violations)
    string(REPLACE ";" "\n  " _pretty "${_violations}")
    message(FATAL_ERROR
      "detrcpp: license scan FAILED — disallowed/unknown licenses in dep graph:\n"
      "  ${_pretty}\n"
      "Update vcpkg.json, or adjust DETR_LICENSE_ALLOWLIST / DETR_LICENSE_DENYLIST.")
  endif()

  message(STATUS
    "detrcpp: license scan passed (${_n_sbom} SBOMs, ${_n_copy} copyright files checked).")
endfunction()
