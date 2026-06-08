// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Threading contract: sink configuration (EnableJsonSink / EnableRemoteSink)
// must happen during single-threaded startup, before any worker threads begin
// logging. SetGlobalLevel is safe at any time (spdlog levels are atomic).
// Loggers are created lazily and live for the whole process; Shutdown() only
// flushes — it never destroys loggers, so references returned by Get() stay
// valid for the program's lifetime.

#include "detr/log/log.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <spdlog/details/log_msg.h>
#include <spdlog/formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace detr::log {

namespace {

// spdlog's memory_buf_t is an fmt buffer: append takes [begin, end), not a
// string_view. This helper makes literal appends readable.
void Raw(spdlog::memory_buf_t& out, std::string_view s) {
  out.append(s.data(), s.data() + s.size());
}

// Appends |data|..|data+size| to |out| with RFC 8259 JSON string escaping.
void AppendJsonEscaped(const char* data, std::size_t size, spdlog::memory_buf_t& out) {
  for (std::size_t i = 0; i < size; ++i) {
    const unsigned char c = static_cast<unsigned char>(data[i]);
    switch (c) {
      case '"':  Raw(out, "\\\""); break;
      case '\\': Raw(out, "\\\\"); break;
      case '\b': Raw(out, "\\b");  break;
      case '\f': Raw(out, "\\f");  break;
      case '\n': Raw(out, "\\n");  break;
      case '\r': Raw(out, "\\r");  break;
      case '\t': Raw(out, "\\t");  break;
      default:
        if (c < 0x20) {
          // Other control characters → \u00XX.
          static constexpr char kHex[] = "0123456789abcdef";
          const char esc[6] = {'\\', 'u', '0', '0',
                               kHex[(c >> 4) & 0xF], kHex[c & 0xF]};
          out.append(esc, esc + sizeof(esc));
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
}

// Emits one NDJSON object per record with all string fields escaped. Far more
// robust than a printf-style %v pattern, which would emit invalid JSON the
// moment a message contained a quote or newline.
class JsonLineFormatter final : public spdlog::formatter {
 public:
  void format(const spdlog::details::log_msg& msg, spdlog::memory_buf_t& dest) override {
    Raw(dest, R"({"level":")");
    const auto level = spdlog::level::to_string_view(msg.level);
    AppendJsonEscaped(level.data(), level.size(), dest);
    Raw(dest, R"(","logger":")");
    AppendJsonEscaped(msg.logger_name.data(), msg.logger_name.size(), dest);
    Raw(dest, R"(","msg":")");
    AppendJsonEscaped(msg.payload.data(), msg.payload.size(), dest);
    Raw(dest, "\"}\n");
  }

  std::unique_ptr<spdlog::formatter> clone() const override {
    return std::make_unique<JsonLineFormatter>();
  }
};

struct LoggerRegistry {
  std::mutex mu;
  std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> loggers;
  std::vector<spdlog::sink_ptr> shared_sinks;
  spdlog::level::level_enum level{spdlog::level::info};

  LoggerRegistry() {
    auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    stderr_sink->set_pattern("[%Y-%m-%dT%H:%M:%S.%e%z] [%n] [%^%l%$] %v");
    shared_sinks.push_back(std::move(stderr_sink));
  }
};

LoggerRegistry& Registry() {
  static LoggerRegistry r;
  return r;
}

spdlog::level::level_enum ToSpd(Level l) {
  switch (l) {
    case Level::Trace:    return spdlog::level::trace;
    case Level::Debug:    return spdlog::level::debug;
    case Level::Info:     return spdlog::level::info;
    case Level::Warn:     return spdlog::level::warn;
    case Level::Error:    return spdlog::level::err;
    case Level::Critical: return spdlog::level::critical;
    case Level::Off:      return spdlog::level::off;
  }
  return spdlog::level::info;
}

}  // namespace

spdlog::logger& Get(std::string_view name) {
  auto& r = Registry();
  std::lock_guard<std::mutex> lock(r.mu);
  std::string key{name};
  auto it = r.loggers.find(key);
  if (it != r.loggers.end()) {
    return *it->second;
  }
  auto lg = std::make_shared<spdlog::logger>(key, r.shared_sinks.begin(), r.shared_sinks.end());
  lg->set_level(r.level);
  lg->flush_on(spdlog::level::warn);
  auto [inserted, _] = r.loggers.emplace(std::move(key), std::move(lg));
  return *inserted->second;
}

void SetGlobalLevel(Level level) {
  auto& r = Registry();
  std::lock_guard<std::mutex> lock(r.mu);
  r.level = ToSpd(level);
  for (auto& [_, lg] : r.loggers) {
    lg->set_level(r.level);  // atomic — safe even concurrently.
  }
}

void EnableJsonSink(const std::string& path) {
  if (path.empty()) {
    return;
  }
  auto& r = Registry();
  std::lock_guard<std::mutex> lock(r.mu);
  auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, /*truncate=*/false);
  sink->set_formatter(std::make_unique<JsonLineFormatter>());
  r.shared_sinks.push_back(sink);
  // Attach to already-created loggers. Must run during single-threaded init
  // (see file header) — mutating a live logger's sink list is not concurrent.
  for (auto& [_, lg] : r.loggers) {
    lg->sinks().push_back(sink);
  }
}

void EnableRemoteSink(const std::string& /*url*/) {
  // Phase 3 — defer wiring an HTTP sink until libcurl is in the dep graph.
  Get("log").warn("remote logging requested but not yet implemented");
}

void Shutdown() {
  auto& r = Registry();
  std::lock_guard<std::mutex> lock(r.mu);
  for (auto& [_, lg] : r.loggers) {
    lg->flush();
  }
  // Deliberately do NOT clear the map: references handed out by Get() must stay
  // valid until process exit. No logging may occur after Shutdown().
}

}  // namespace detr::log
