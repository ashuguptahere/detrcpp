// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/pth.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "detr/weights/legacy_pickle.hpp"
#include "detr/weights/tensor.hpp"
#include "detr/weights/zip.hpp"
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

// The archive prefix (everything before "data.pkl", e.g. "archive/") for a torch.save zip.
Result<std::string> FindPrefix(const ZipReader& zip) {
  constexpr std::string_view kPkl = "data.pkl";
  for (const std::string& n : zip.Names()) {
    if (n.size() >= kPkl.size() && n.compare(n.size() - kPkl.size(), kPkl.size(), kPkl) == 0) {
      const std::size_t pre = n.size() - kPkl.size();
      if (pre == 0 || n[pre - 1] == '/') return n.substr(0, pre);
    }
  }
  return Err(ErrorCode::ParseError, ".pth archive has no data.pkl entry");
}

// The (name -> tensor) pairs: the top-level dict, or — for a wrapped training
// checkpoint — the nested weights. Recurses so multi-level wrappers resolve (e.g.
// the EMA checkpoints these repos ship as {"ema": {"module": <weights>, ...}}).
// A canonical child name ("model"/"state_dict"/"ema"/"module") is preferred over the
// merely-richest dict, so optimizer state never wins over the model weights.
std::vector<std::pair<std::string, Value>> PickTensors(const Value& root) {
  std::vector<std::pair<std::string, Value>> tensors = NamedTensors(root);
  if (!tensors.empty()) return tensors;
  std::size_t best = 0;
  for (const auto& [k, v] : root.entries) {
    if (v.kind != Value::Kind::Dict) continue;
    auto nested = PickTensors(v);  // recurse: handles ema.module and deeper wrappers
    const bool canonical = k.kind == Value::Kind::Str &&
                           (k.s == "model" || k.s == "state_dict" || k.s == "model_state_dict" ||
                            k.s == "ema" || k.s == "module");
    if (canonical && !nested.empty()) return nested;
    if (nested.size() > best) {
      best = nested.size();
      tensors = std::move(nested);
    }
  }
  return tensors;
}

// Reads a modern torch.save zip (data.pkl + data/<key> storages) into a StateDict.
Result<StateDict> LoadModern(const ZipReader& zip) {
  auto prefix = FindPrefix(zip);
  if (!prefix) return Err(prefix.error().code, prefix.error().message);
  auto pkl = zip.Read(*prefix + "data.pkl");
  if (!pkl) return Err(pkl.error().code, pkl.error().message);

  Unpickler up(std::span<const std::byte>(pkl->data(), pkl->size()), 0);
  auto root = up.Run();
  if (!root) return Err(root.error().code, root.error().message);
  if (root->kind != Value::Kind::Dict) return Err(ErrorCode::Unsupported, ".pth root is not a state_dict");

  auto tensors = PickTensors(*root);
  if (tensors.empty()) return Err(ErrorCode::Unsupported, "no tensors recovered from .pth");

  StateDict sd;
  for (const auto& [name, t] : tensors) {
    std::int64_t prod = 1;
    for (const std::int64_t d : t.shape) {
      if (d < 0 || prod > kMaxItems) {
        return Err(ErrorCode::ParseError, fmt::format(".pth tensor '{}' bad shape", name));
      }
      prod *= d;
    }
    auto storage = zip.Read(*prefix + "data/" + t.s);
    if (!storage) return Err(ErrorCode::ParseError, fmt::format(".pth tensor '{}' missing storage", name));
    const std::size_t elsize = DTypeSize(t.dt);
    const std::size_t span_bytes = static_cast<std::size_t>(prod) * elsize;
    RawTensor rt;
    rt.dtype = t.dt;
    rt.shape = t.shape;
    if (detail::IsContiguous(t.shape, t.stride)) {
      const std::size_t begin = static_cast<std::size_t>(t.i) * elsize;
      if (begin + span_bytes > storage->size()) {
        return Err(ErrorCode::ParseError, fmt::format(".pth tensor '{}' slice out of range", name));
      }
      rt.data.assign(storage->begin() + static_cast<std::ptrdiff_t>(begin),
                     storage->begin() + static_cast<std::ptrdiff_t>(begin + span_bytes));
    } else {
      // Non-contiguous storage (e.g. a transposed/permuted weight saved as a view):
      // gather elements into C-contiguous row-major order per (storage_offset, stride).
      const auto nd = t.shape.size();
      rt.data.resize(span_bytes);
      std::vector<std::int64_t> idx(nd, 0);
      for (std::int64_t lin = 0; lin < prod; ++lin) {
        std::int64_t src = t.i;  // storage_offset, in elements
        for (std::size_t k = 0; k < nd; ++k) src += idx[k] * t.stride[k];
        const std::size_t src_bytes = static_cast<std::size_t>(src) * elsize;
        if (src < 0 || src_bytes + elsize > storage->size()) {
          return Err(ErrorCode::ParseError, fmt::format(".pth tensor '{}' stride out of range", name));
        }
        std::copy(storage->begin() + static_cast<std::ptrdiff_t>(src_bytes),
                  storage->begin() + static_cast<std::ptrdiff_t>(src_bytes + elsize),
                  rt.data.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(lin) * elsize));
        for (std::size_t k = nd; k-- > 0;) {  // row-major increment, last dim fastest
          if (++idx[k] < t.shape[k]) break;
          idx[k] = 0;
        }
      }
    }
    sd.Set(name, std::move(rt));
  }
  return sd;
}

// ---- writer ----

// The torch.Storage class name pickled in a persistent_id for each dtype.
std::string_view StorageTypeName(DType dt) {
  switch (dt) {
    case DType::F64: return "DoubleStorage";
    case DType::F32: return "FloatStorage";
    case DType::F16: return "HalfStorage";
    case DType::BF16: return "BFloat16Storage";
    case DType::I64: return "LongStorage";
    case DType::I32: return "IntStorage";
    case DType::I16: return "ShortStorage";
    case DType::I8: return "CharStorage";
    case DType::U8: return "ByteStorage";
    case DType::Bool: return "BoolStorage";
  }
  return "FloatStorage";
}

// A tiny protocol-2 pickle emitter (no memoization — we never back-reference).
struct PickleWriter {
  std::vector<std::byte> out;
  void Byte(std::uint8_t b) { out.push_back(static_cast<std::byte>(b)); }
  void Raw(const void* p, std::size_t n) {
    const auto* b = static_cast<const std::byte*>(p);
    out.insert(out.end(), b, b + n);
  }
  void U32le(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) Byte(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
  }
  void Str(std::string_view s) {  // BINUNICODE
    Byte(0x58);
    U32le(static_cast<std::uint32_t>(s.size()));
    Raw(s.data(), s.size());
  }
  void Global(std::string_view module, std::string_view name) {  // GLOBAL module\nname\n
    Byte(0x63);
    Raw(module.data(), module.size());
    Byte('\n');
    Raw(name.data(), name.size());
    Byte('\n');
  }
  void Int(std::int64_t n) {
    if (n >= 0 && n <= 0xff) {
      Byte(0x4b);  // BININT1
      Byte(static_cast<std::uint8_t>(n));
    } else if (n >= 0 && n <= 0xffff) {
      Byte(0x4d);  // BININT2
      Byte(static_cast<std::uint8_t>(n & 0xff));
      Byte(static_cast<std::uint8_t>((n >> 8) & 0xff));
    } else if (n >= -0x80000000LL && n <= 0x7fffffffLL) {
      Byte(0x4a);  // BININT (4-byte signed)
      U32le(static_cast<std::uint32_t>(static_cast<std::int32_t>(n)));
    } else {
      // LONG1: minimal little-endian two's-complement bytes.
      std::vector<std::uint8_t> b;
      std::uint64_t u = static_cast<std::uint64_t>(n);
      for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>((u >> (8 * i)) & 0xff));
      while (b.size() > 1 && b.back() == 0x00 && (b[b.size() - 2] & 0x80) == 0) b.pop_back();
      Byte(0x8a);  // LONG1
      Byte(static_cast<std::uint8_t>(b.size()));
      for (std::uint8_t x : b) Byte(x);
    }
  }
  void Mark() { Byte(0x28); }
  void Tuple() { Byte(0x74); }       // TUPLE (collect to MARK)
  void EmptyDict() { Byte(0x7d); }   // EMPTY_DICT
  void EmptyTuple() { Byte(0x29); }  // EMPTY_TUPLE
  void NewFalse() { Byte(0x89); }
  void Persid() { Byte(0x51); }   // BINPERSID
  void Reduce() { Byte(0x52); }
  void SetItems() { Byte(0x75); }
  void Stop() { Byte(0x2e); }
  void IntTuple(const std::vector<std::int64_t>& v) {
    Mark();
    for (std::int64_t x : v) Int(x);
    Tuple();
  }
};

}  // namespace

Result<void> SavePth(const std::filesystem::path& path, const StateDict& state) {
  const std::vector<std::string>& keys = state.Keys();
  // Build data.pkl: a plain dict { name -> _rebuild_tensor_v2((storage), 0, size, stride,
  // False, {}) }, where each storage is the i-th `data/<i>` zip entry. Matches torch.save.
  PickleWriter pw;
  pw.Byte(0x80);
  pw.Byte(0x02);  // PROTO 2
  pw.EmptyDict();
  pw.Mark();
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const RawTensor* rt = state.Find(keys[i]);
    std::int64_t numel = 1;
    for (std::int64_t d : rt->shape) numel *= d;
    // C-contiguous strides for this shape.
    std::vector<std::int64_t> stride(rt->shape.size());
    std::int64_t acc = 1;
    for (std::size_t k = rt->shape.size(); k-- > 0;) {
      stride[k] = acc;
      acc *= rt->shape[k];
    }
    pw.Str(keys[i]);                                       // dict key
    pw.Global("torch._utils", "_rebuild_tensor_v2");
    pw.Mark();                                             // args tuple
    pw.Mark();                                             // persistent_id tuple
    pw.Str("storage");
    pw.Global("torch", StorageTypeName(rt->dtype));
    pw.Str(std::to_string(i));                             // storage key -> data/<i>
    pw.Str("cpu");
    pw.Int(numel);
    pw.Tuple();
    pw.Persid();
    pw.Int(0);                                             // storage_offset
    pw.IntTuple(rt->shape);                                // size
    pw.IntTuple(stride);                                   // stride
    pw.NewFalse();                                         // requires_grad
    pw.EmptyDict();                                        // backward_hooks
    pw.Tuple();
    pw.Reduce();
  }
  pw.SetItems();
  pw.Stop();

  const std::string stem = path.stem().string();
  auto w = ZipWriter::Create(path);
  if (!w) return Err(w.error().code, w.error().message);
  auto add = [&](const std::string& name, std::span<const std::byte> data) {
    return w->Add(stem + "/" + name, data);
  };
  if (auto r = add("data.pkl", pw.out); !r) return r;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const RawTensor* rt = state.Find(keys[i]);
    if (auto r = add("data/" + std::to_string(i), rt->data); !r) return r;
  }
  const std::string ver = "3\n";
  const std::string bo = "little";
  if (auto r = add("version", std::as_bytes(std::span<const char>(ver.data(), ver.size()))); !r) return r;
  if (auto r = add("byteorder", std::as_bytes(std::span<const char>(bo.data(), bo.size()))); !r) return r;
  return w->Finish();
}

// Reads a PaddlePaddle `.pdparams`: a single pickle whose values are inline numpy arrays
// (no zip, no persistent_id storages). Returns Unsupported when the pickle yields no
// inline-data tensors (i.e. it is a legacy torch pickle, handled elsewhere).
Result<StateDict> LoadPdparams(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Err(ErrorCode::Io, fmt::format("cannot open '{}'", path.string()));
  std::vector<std::byte> buf;
  {
    f.seekg(0, std::ios::end);
    const auto n = f.tellg();
    if (n <= 0) return Err(ErrorCode::ParseError, ".pdparams is empty");
    buf.resize(static_cast<std::size_t>(n));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  }
  Unpickler up(std::span<const std::byte>(buf.data(), buf.size()), 0);
  auto root = up.Run();
  if (!root) return Err(root.error().code, root.error().message);
  if (root->kind != Value::Kind::Dict) return Err(ErrorCode::Unsupported, ".pdparams not a dict");

  StateDict sd;
  for (const auto& [name, t] : PickTensors(*root)) {
    if (!t.inline_bytes) continue;  // a torch storage-ref tensor: not paddle
    std::int64_t prod = 1;
    for (const std::int64_t d : t.shape) {
      if (d < 0 || prod > kMaxItems) return Err(ErrorCode::ParseError, ".pdparams bad shape");
      prod *= d;
    }
    const std::size_t need = static_cast<std::size_t>(prod) * DTypeSize(t.dt);
    if (t.s.size() != need) {
      return Err(ErrorCode::ParseError, fmt::format(".pdparams tensor '{}' data size mismatch", name));
    }
    RawTensor rt;
    rt.dtype = t.dt;
    rt.shape = t.shape;
    const auto* p = reinterpret_cast<const std::byte*>(t.s.data());
    rt.data.assign(p, p + t.s.size());
    sd.Set(name, std::move(rt));
  }
  if (sd.Size() == 0) return Err(ErrorCode::Unsupported, "no inline tensors (not a .pdparams)");
  return sd;
}

Result<StateDict> LoadPth(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Err(ErrorCode::Io, fmt::format("cannot open '{}'", path.string()));
  char magic[2] = {0, 0};
  f.read(magic, 2);
  // A "PK" local-file-header is a modern torch.save zip; 0x80 is a pickle PROTO — which is
  // either a paddle `.pdparams` (inline numpy arrays) or a legacy torch pickle.
  if (magic[0] == 'P' && magic[1] == 'K') {
    auto zip = ZipReader::Open(path);
    if (!zip) return Err(zip.error().code, zip.error().message);
    return LoadModern(*zip);
  }
  if (auto pd = LoadPdparams(path); pd.has_value()) return pd;
  return LoadLegacyPth(path);
}

}  // namespace detr::weights
