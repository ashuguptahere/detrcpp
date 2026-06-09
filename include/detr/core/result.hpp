// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Result<T> is a typedef for tl::expected<T, Error> — the project-wide return
// type for recoverable failures. We use tl::expected rather than std::expected
// because we target C++20 toolchains; when C++23 is universally available the
// typedef will switch to std::expected without touching callers.

#pragma once

#include <string>
#include <string_view>
#include <tl/expected.hpp>
#include <utility>

namespace detr::core {

enum class ErrorCode {
  Ok = 0,
  InvalidArgument,
  NotFound,
  Io,
  ParseError,
  Unsupported,
  PermissionDenied,
  OutOfMemory,
  DeviceUnavailable,
  Internal,
};

struct Error {
  ErrorCode code{ErrorCode::Internal};
  std::string message;

  Error() = default;
  Error(ErrorCode c, std::string m) : code(c), message(std::move(m)) {}
};

template <typename T>
using Result = tl::expected<T, Error>;

inline tl::unexpected<Error> Err(ErrorCode code, std::string message) {
  return tl::make_unexpected(Error{code, std::move(message)});
}

inline std::string_view ToString(ErrorCode c) {
  switch (c) {
    case ErrorCode::Ok:
      return "Ok";
    case ErrorCode::InvalidArgument:
      return "InvalidArgument";
    case ErrorCode::NotFound:
      return "NotFound";
    case ErrorCode::Io:
      return "Io";
    case ErrorCode::ParseError:
      return "ParseError";
    case ErrorCode::Unsupported:
      return "Unsupported";
    case ErrorCode::PermissionDenied:
      return "PermissionDenied";
    case ErrorCode::OutOfMemory:
      return "OutOfMemory";
    case ErrorCode::DeviceUnavailable:
      return "DeviceUnavailable";
    case ErrorCode::Internal:
      return "Internal";
  }
  return "Unknown";
}

}  // namespace detr::core
