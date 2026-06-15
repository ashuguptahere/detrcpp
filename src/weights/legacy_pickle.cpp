// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/legacy_pickle.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "detr/weights/tensor.hpp"
#include "pickle_vm.hpp"

namespace detr::weights {

namespace {

using core::Err;
using core::ErrorCode;
using core::Result;
using detail::kMaxItems;
using detail::NamedTensors;
using detail::Unpickler;
using detail::Value;

std::int64_t ReadI64Le(std::span<const std::byte> b, std::size_t off) {
  std::uint64_t v = 0;
  for (std::size_t k = 0; k < 8; ++k) {
    v |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(b[off + k])) << (8 * k);
  }
  return static_cast<std::int64_t>(v);
}

}  // namespace

Result<StateDict> LoadLegacyPthBytes(std::span<const std::byte> bytes) {
  // Five concatenated pickle archives: magic, protocol, sys_info, the state_dict,
  // and the storage-key list — then the raw storage payloads.
  Unpickler a1(bytes, 0);
  auto magic = a1.Run();
  if (!magic) return Err(magic.error().code, magic.error().message);
  Unpickler a2(bytes, a1.pos());
  auto proto = a2.Run();
  if (!proto) return Err(proto.error().code, proto.error().message);
  Unpickler a3(bytes, a2.pos());
  auto sys = a3.Run();
  if (!sys) return Err(sys.error().code, sys.error().message);
  if (sys->kind == Value::Kind::Dict) {
    for (const auto& [k, v] : sys->entries) {
      if (k.kind == Value::Kind::Str && k.s == "little_endian" && v.kind == Value::Kind::Bool &&
          !v.b) {
        return Err(ErrorCode::Unsupported, "legacy checkpoint is big-endian (unsupported)");
      }
    }
  }

  Unpickler a4(bytes, a3.pos());
  auto main = a4.Run();
  if (!main) return Err(main.error().code, main.error().message);
  if (main->kind != Value::Kind::Dict) {
    return Err(ErrorCode::Unsupported, "legacy checkpoint root is not a state_dict");
  }

  Unpickler a5(bytes, a4.pos());
  auto keylist = a5.Run();
  if (!keylist) return Err(keylist.error().code, keylist.error().message);
  std::vector<std::string> keys;
  for (const Value& e : keylist->items) {
    if (e.kind == Value::Kind::Str) {
      keys.push_back(e.s);
    } else if (e.kind == Value::Kind::Int) {
      keys.push_back(std::to_string(e.i));
    } else {
      return Err(ErrorCode::Unsupported, "legacy storage-key list has a non-string key");
    }
  }

  // Output tensors: the top-level state_dict, or — for a training checkpoint that
  // wraps it — the nested "model"/"state_dict" (else the richest nested dict).
  std::vector<std::pair<std::string, Value>> tensors = NamedTensors(*main);
  if (tensors.empty()) {
    std::size_t best = 0;
    for (const auto& [k, v] : main->entries) {
      if (v.kind != Value::Kind::Dict) continue;
      auto nested = NamedTensors(v);
      const bool canonical = k.kind == Value::Kind::Str &&
                             (k.s == "model" || k.s == "state_dict" || k.s == "model_state_dict");
      if (canonical && !nested.empty()) {
        tensors = std::move(nested);
        break;
      }
      if (nested.size() > best) {
        best = nested.size();
        tensors = std::move(nested);
      }
    }
  }
  // The storage-key list covers EVERY storage, so use the dtypes recorded at every
  // persistent_id during parsing (more complete than walking only output tensors).
  std::map<std::string, DType> key_dt = a4.storages();

  // Walk the storage section: for each key, an int64 element count then the bytes.
  std::size_t sp = a5.pos();
  std::map<std::string, std::vector<std::byte>> store;
  for (const std::string& key : keys) {
    if (sp + 8 > bytes.size()) return Err(ErrorCode::ParseError, "legacy storage: truncated size");
    const std::int64_t numel = ReadI64Le(bytes, sp);
    sp += 8;
    const auto dt = key_dt.find(key);
    if (dt == key_dt.end()) {
      return Err(ErrorCode::Unsupported, fmt::format("legacy storage '{}' has no tensor", key));
    }
    if (numel < 0 || numel > kMaxItems) {
      return Err(ErrorCode::ParseError, "legacy storage: implausible element count");
    }
    const std::size_t nbytes = static_cast<std::size_t>(numel) * DTypeSize(dt->second);
    if (sp + nbytes > bytes.size()) return Err(ErrorCode::ParseError, "legacy storage: truncated data");
    store[key].assign(bytes.begin() + static_cast<std::ptrdiff_t>(sp),
                      bytes.begin() + static_cast<std::ptrdiff_t>(sp + nbytes));
    sp += nbytes;
  }

  StateDict sd;
  for (const auto& [name, t] : tensors) {
    if (!detail::IsContiguous(t.shape, t.stride)) {
      return Err(ErrorCode::Unsupported, fmt::format("legacy tensor '{}' is non-contiguous", name));
    }
    std::int64_t prod = 1;
    for (const std::int64_t d : t.shape) {
      if (d < 0 || prod > kMaxItems) {
        return Err(ErrorCode::ParseError, fmt::format("legacy tensor '{}' bad shape", name));
      }
      prod *= d;
    }
    const auto sit = store.find(t.s);
    if (sit == store.end()) {
      return Err(ErrorCode::ParseError, fmt::format("legacy tensor '{}' missing storage", name));
    }
    const std::size_t elsize = DTypeSize(t.dt);
    const std::size_t begin = static_cast<std::size_t>(t.i) * elsize;
    const std::size_t span_bytes = static_cast<std::size_t>(prod) * elsize;
    if (begin + span_bytes > sit->second.size()) {
      return Err(ErrorCode::ParseError, fmt::format("legacy tensor '{}' slice out of range", name));
    }
    RawTensor rt;
    rt.dtype = t.dt;
    rt.shape = t.shape;
    rt.data.assign(sit->second.begin() + static_cast<std::ptrdiff_t>(begin),
                   sit->second.begin() + static_cast<std::ptrdiff_t>(begin + span_bytes));
    sd.Set(name, std::move(rt));
  }
  if (sd.Empty()) return Err(ErrorCode::Unsupported, "no tensors recovered from legacy checkpoint");
  return sd;
}

Result<StateDict> LoadLegacyPth(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return Err(ErrorCode::Io, fmt::format("cannot open '{}'", path.string()));
  const auto size = f.tellg();
  if (size < 0) return Err(ErrorCode::Io, fmt::format("cannot size '{}'", path.string()));
  std::vector<std::byte> buf(static_cast<std::size_t>(size));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(buf.data()), size);
  if (!f) return Err(ErrorCode::Io, fmt::format("cannot read '{}'", path.string()));
  return LoadLegacyPthBytes(buf);
}

}  // namespace detr::weights
