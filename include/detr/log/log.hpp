// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Hierarchical logging façade over spdlog. All logging in the codebase goes
// through this header — never include <spdlog/...> directly outside src/log/.
// Loggers are addressed by dotted names ("train.optimizer", "infer.trt"); each
// is created lazily on first Get(). Output: stderr (human) plus, when enabled,
// an NDJSON sink consumed by the GUI and external tooling.

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include <spdlog/spdlog.h>

namespace detr::log {

enum class Level {
  Trace,
  Debug,
  Info,
  Warn,
  Error,
  Critical,
  Off,
};

// Returns the logger registered under |name|, creating it if necessary.
// Thread-safe.
spdlog::logger& Get(std::string_view name);

// Sets the level on every existing and future logger. Default Info.
void SetGlobalLevel(Level level);

// Enables a JSON-lines sink writing one event per line to |path|. Used by
// `runs/<exp>/log.ndjson`. Pass an empty path to disable.
void EnableJsonSink(const std::string& path);

// Optional remote sink (URL must be http(s)://). Disabled by default.
void EnableRemoteSink(const std::string& url);

// Shuts down all sinks and flushes pending events. Call before exit().
void Shutdown();

}  // namespace detr::log
