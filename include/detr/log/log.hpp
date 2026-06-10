// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Hierarchical logging façade. All logging in the codebase goes through this
// header; spdlog is an implementation detail confined to src/log/ and never
// leaks into another translation unit. Loggers are addressed by dotted names
// ("train.optimizer", "infer.trt") and created lazily on first Get(). Output:
// stderr (human) plus, when enabled, an NDJSON sink consumed by the GUI and
// external tooling. Format strings are compile-time checked (like fmt/spdlog).

#pragma once

#include <fmt/format.h>

#include <string>
#include <string_view>
#include <utility>

namespace detr::log {

enum class Level { Trace, Debug, Info, Warn, Error, Critical, Off };

namespace detail {
// Implemented in src/log/log.cpp over spdlog; |handle| identifies the logger.
void Emit(void* handle, Level level, std::string_view msg);
bool ShouldLog(const void* handle, Level level);
}  // namespace detail

// A cheap, copyable handle to a named logger; all state lives in the registry.
// The format-string overloads format with fmt at the call site and forward the
// finished message to the sink, so spdlog never appears in a caller's TU.
class Logger {
 public:
  explicit Logger(void* handle = nullptr) : handle_(handle) {}

  template <typename... Args>
  void trace(fmt::format_string<Args...> f, Args&&... args) const {
    Log(Level::Trace, f, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void debug(fmt::format_string<Args...> f, Args&&... args) const {
    Log(Level::Debug, f, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void info(fmt::format_string<Args...> f, Args&&... args) const {
    Log(Level::Info, f, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void warn(fmt::format_string<Args...> f, Args&&... args) const {
    Log(Level::Warn, f, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void error(fmt::format_string<Args...> f, Args&&... args) const {
    Log(Level::Error, f, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void critical(fmt::format_string<Args...> f, Args&&... args) const {
    Log(Level::Critical, f, std::forward<Args>(args)...);
  }

  bool ShouldLog(Level level) const { return detail::ShouldLog(handle_, level); }

 private:
  template <typename... Args>
  void Log(Level level, fmt::format_string<Args...> f, Args&&... args) const {
    if (!detail::ShouldLog(handle_, level)) {
      return;
    }
    detail::Emit(handle_, level, fmt::format(f, std::forward<Args>(args)...));
  }

  void* handle_;
};

// Returns the logger registered under |name|, creating it if necessary.
// Thread-safe; the reference is stable for the process lifetime.
Logger& Get(std::string_view name);

// Sets the level on every existing and future logger. Default Info.
void SetGlobalLevel(Level level);

// Enables a JSON-lines sink writing one event per line to |path| (used by
// `runs/<exp>/log.ndjson`). Pass an empty path to disable.
void EnableJsonSink(const std::string& path);

// Optional remote sink (URL must be http(s)://). Disabled by default.
void EnableRemoteSink(const std::string& url);

// Flushes all sinks. Call before exit(); no logging may occur afterwards.
void Shutdown();

}  // namespace detr::log
