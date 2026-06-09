// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/safetensors.hpp"

#include <fmt/format.h>
#include <simdjson.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "detr/weights/tensor.hpp"

namespace detr::weights {

namespace {

using core::Err;
using core::ErrorCode;
using core::Result;

// Defensive upper bound on the JSON header so a corrupt length cannot trigger a
// huge allocation. Real model headers are kilobytes; 256 MiB is absurdly safe.
constexpr std::uint64_t kMaxHeaderBytes = 256ull * 1024 * 1024;

std::uint64_t ReadLittleEndianU64(const char* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i);
  }
  return v;
}

void AppendJsonEscaped(std::string_view s, std::string& out) {
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          static constexpr char kHex[] = "0123456789abcdef";
          out += "\\u00";
          out += kHex[(c >> 4) & 0xF];
          out += kHex[c & 0xF];
        } else {
          out += static_cast<char>(c);
        }
    }
  }
}

}  // namespace

Result<StateDict> LoadSafetensors(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return Err(ErrorCode::Io, fmt::format("cannot open '{}'", path.string()));
  }
  const std::streamoff end = f.tellg();
  if (end < 8) {
    return Err(ErrorCode::ParseError,
               fmt::format("'{}' too small to be safetensors", path.string()));
  }
  const auto file_size = static_cast<std::uint64_t>(end);

  std::vector<char> buf(static_cast<std::size_t>(file_size));
  f.seekg(0);
  if (!f.read(buf.data(), static_cast<std::streamsize>(file_size))) {
    return Err(ErrorCode::Io, fmt::format("short read on '{}'", path.string()));
  }

  const std::uint64_t header_len = ReadLittleEndianU64(buf.data());
  if (header_len > kMaxHeaderBytes || 8 + header_len > file_size) {
    return Err(ErrorCode::ParseError,
               fmt::format("bad safetensors header length {} in '{}'", header_len, path.string()));
  }

  const char* data_begin = buf.data() + 8 + header_len;
  const std::uint64_t data_len = file_size - 8 - header_len;

  simdjson::dom::parser parser;
  simdjson::dom::element doc;
  auto perr = parser.parse(buf.data() + 8, static_cast<std::size_t>(header_len)).get(doc);
  if (perr) {
    return Err(ErrorCode::ParseError, fmt::format("header JSON parse error in '{}': {}",
                                                  path.string(), simdjson::error_message(perr)));
  }
  simdjson::dom::object root;
  if (doc.get(root)) {
    return Err(ErrorCode::ParseError, "safetensors header is not a JSON object");
  }

  StateDict out;
  for (auto field : root) {
    const std::string_view key = field.key;

    if (key == "__metadata__") {
      simdjson::dom::object meta;
      if (field.value.get(meta) == simdjson::SUCCESS) {
        for (auto m : meta) {
          std::string_view mv;
          if (m.value.get_string().get(mv) == simdjson::SUCCESS) {
            out.SetMeta(std::string(m.key), std::string(mv));
          }
        }
      }
      continue;
    }

    simdjson::dom::object t;
    if (field.value.get(t)) {
      return Err(ErrorCode::ParseError, fmt::format("tensor '{}' is not an object", key));
    }

    std::string_view dtype_s;
    if (t["dtype"].get_string().get(dtype_s)) {
      return Err(ErrorCode::ParseError, fmt::format("tensor '{}' missing dtype", key));
    }
    auto dtype = DTypeFromName(dtype_s);
    if (!dtype) {
      return tl::make_unexpected(dtype.error());
    }

    simdjson::dom::array shape_a;
    if (t["shape"].get_array().get(shape_a)) {
      return Err(ErrorCode::ParseError, fmt::format("tensor '{}' missing shape", key));
    }
    RawTensor rt;
    rt.dtype = *dtype;
    std::int64_t numel = 1;
    for (auto dim : shape_a) {
      std::int64_t d = 0;
      if (dim.get_int64().get(d) || d < 0) {
        return Err(ErrorCode::ParseError, fmt::format("tensor '{}' bad shape dim", key));
      }
      rt.shape.push_back(d);
      numel *= d;
    }
    if (rt.shape.empty()) {
      numel = 1;  // rank-0 scalar
    }

    simdjson::dom::array off_a;
    if (t["data_offsets"].get_array().get(off_a)) {
      return Err(ErrorCode::ParseError, fmt::format("tensor '{}' missing data_offsets", key));
    }
    std::vector<std::int64_t> offs;
    for (auto o : off_a) {
      std::int64_t v = 0;
      if (o.get_int64().get(v) || v < 0) {
        return Err(ErrorCode::ParseError, fmt::format("tensor '{}' bad data_offsets", key));
      }
      offs.push_back(v);
    }
    if (offs.size() != 2 || offs[0] > offs[1] || static_cast<std::uint64_t>(offs[1]) > data_len) {
      return Err(ErrorCode::ParseError, fmt::format("tensor '{}' data_offsets out of range", key));
    }

    const std::uint64_t span = static_cast<std::uint64_t>(offs[1] - offs[0]);
    const std::uint64_t expected = static_cast<std::uint64_t>(numel) * DTypeSize(*dtype);
    if (span != expected) {
      return Err(ErrorCode::ParseError,
                 fmt::format("tensor '{}' byte span {} != shape*dtype {}", key, span, expected));
    }

    rt.data.resize(static_cast<std::size_t>(span));
    const char* src = data_begin + offs[0];
    std::memcpy(rt.data.data(), src, static_cast<std::size_t>(span));
    out.Set(std::string(key), std::move(rt));
  }

  return out;
}

Result<void> SaveSafetensors(const std::filesystem::path& path, const StateDict& state) {
  // Validate every tensor's byte length before we write anything.
  for (const auto& name : state.Keys()) {
    const RawTensor* t = state.Find(name);
    if (t->data.size() != t->Nbytes()) {
      return Err(ErrorCode::InvalidArgument,
                 fmt::format("tensor '{}' data {} bytes != shape*dtype {} bytes", name,
                             t->data.size(), t->Nbytes()));
    }
  }

  // Build header JSON and compute contiguous data offsets (insertion order).
  std::string header = "{";
  bool first = true;
  std::uint64_t offset = 0;
  for (const auto& name : state.Keys()) {
    const RawTensor* t = state.Find(name);
    if (!first) {
      header += ',';
    }
    first = false;
    header += '"';
    AppendJsonEscaped(name, header);
    header += "\":{\"dtype\":\"";
    header += DTypeName(t->dtype);
    header += "\",\"shape\":[";
    for (std::size_t i = 0; i < t->shape.size(); ++i) {
      if (i) {
        header += ',';
      }
      header += std::to_string(t->shape[i]);
    }
    const std::uint64_t begin = offset;
    offset += t->data.size();
    header += fmt::format("],\"data_offsets\":[{},{}]}}", begin, offset);
  }
  if (!state.Meta().empty()) {
    header += ",\"__metadata__\":{";
    bool mfirst = true;
    for (const auto& [k, v] : state.Meta()) {
      if (!mfirst) {
        header += ',';
      }
      mfirst = false;
      header += '"';
      AppendJsonEscaped(k, header);
      header += "\":\"";
      AppendJsonEscaped(v, header);
      header += '"';
    }
    header += '}';
  }
  header += '}';

  // Pad the header with spaces so the data buffer is 8-byte aligned (matches the
  // reference safetensors writer; trailing whitespace is valid per the spec).
  while ((8 + header.size()) % 8 != 0) {
    header += ' ';
  }

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    return Err(ErrorCode::Io, fmt::format("cannot write '{}'", path.string()));
  }
  char len_le[8];
  const std::uint64_t hlen = header.size();
  for (int i = 0; i < 8; ++i) {
    len_le[i] = static_cast<char>((hlen >> (8 * i)) & 0xFF);
  }
  f.write(len_le, 8);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const auto& name : state.Keys()) {
    const RawTensor* t = state.Find(name);
    f.write(reinterpret_cast<const char*>(t->data.data()),
            static_cast<std::streamsize>(t->data.size()));
  }
  if (!f) {
    return Err(ErrorCode::Io, fmt::format("write failed on '{}'", path.string()));
  }
  return {};
}

}  // namespace detr::weights
