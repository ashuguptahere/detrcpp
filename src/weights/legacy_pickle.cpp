// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/legacy_pickle.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "detr/weights/tensor.hpp"

namespace detr::weights {

namespace {

using core::Err;
using core::ErrorCode;
using core::Result;

constexpr std::int64_t kMaxItems = 1'000'000'000;  // sanity cap on numel / dims

std::optional<DType> StorageDType(std::string_view name) {
  if (name == "FloatStorage") return DType::F32;
  if (name == "DoubleStorage") return DType::F64;
  if (name == "HalfStorage") return DType::F16;
  if (name == "BFloat16Storage") return DType::BF16;
  if (name == "LongStorage") return DType::I64;
  if (name == "IntStorage") return DType::I32;
  if (name == "ShortStorage") return DType::I16;
  if (name == "CharStorage") return DType::I8;
  if (name == "ByteStorage") return DType::U8;
  if (name == "BoolStorage") return DType::Bool;
  return std::nullopt;
}

// A value on the pickle stack. Recognized torch constructs (Storage/Tensor) carry
// the descriptor we actually need; everything else is opaque stack manipulation.
struct Value {
  enum class Kind { None, Bool, Int, Str, Tuple, Dict, Mark, Global, Storage, Tensor, Opaque };
  Kind kind{Kind::None};
  bool b{false};
  std::int64_t i{0};  // Int value, or Tensor storage_offset
  std::string s;      // Str; Global module; Storage/Tensor storage key
  std::string s2;     // Global name
  std::vector<Value> items;                      // Tuple elements
  std::vector<std::pair<Value, Value>> entries;  // Dict
  DType dt{DType::F32};                           // Storage/Tensor dtype
  std::int64_t numel{0};                          // Storage element count
  std::vector<std::int64_t> shape;               // Tensor size
  std::vector<std::int64_t> stride;              // Tensor stride
};

// A minimal protocol-2 pickle interpreter. It never dispatches an unknown global
// (no code execution); unrecognized reduces become inert Opaque values. The memo
// holds value copies, which is correct here because a state_dict never references
// its mutated root container again. Each Run() consumes one archive up to STOP.
class Unpickler {
 public:
  Unpickler(std::span<const std::byte> buf, std::size_t pos) : buf_(buf), pos_(pos) {}
  std::size_t pos() const { return pos_; }

  Result<Value> Run() {
    while (true) {
      const std::uint8_t op = U8();
      if (err_) {
        return Err(code_, msg_);
      }
      switch (op) {
        case 0x80: U8(); break;                                    // PROTO (skip version)
        case 0x7d: Push(MakeKind(Value::Kind::Dict)); break;       // EMPTY_DICT
        case 0x29: Push(MakeKind(Value::Kind::Tuple)); break;      // EMPTY_TUPLE
        case 0x5d: Push(MakeKind(Value::Kind::Tuple)); break;      // EMPTY_LIST (as tuple)
        case 0x28: Push(MakeKind(Value::Kind::Mark)); break;       // MARK
        case 0x71: MemoPut(U8()); break;                           // BINPUT
        case 0x72: MemoPut(U32()); break;                          // LONG_BINPUT
        case 0x68: MemoGet(U8()); break;                           // BINGET
        case 0x6a: MemoGet(U32()); break;                          // LONG_BINGET
        case 0x58: Push(MakeStr(Bytes(U32()))); break;             // BINUNICODE
        case 0x8c: Push(MakeStr(Bytes(U8()))); break;              // SHORT_BINUNICODE
        case 0x4a: Push(MakeInt(static_cast<std::int64_t>(I32()))); break;  // BININT
        case 0x4b: Push(MakeInt(static_cast<std::int64_t>(U8()))); break;   // BININT1
        case 0x4d: Push(MakeInt(static_cast<std::int64_t>(U16()))); break;  // BININT2
        case 0x8a: Push(MakeInt(LongN(U8()))); break;              // LONG1
        case 0x8b: Push(MakeInt(LongN(U32()))); break;             // LONG4
        case 0x88: Push(MakeBool(true)); break;                    // NEWTRUE
        case 0x89: Push(MakeBool(false)); break;                   // NEWFALSE
        case 0x4e: Push(MakeKind(Value::Kind::None)); break;       // NONE
        case 0x74: MakeTupleFromMark(); break;                     // TUPLE
        case 0x85: MakeTupleN(1); break;                           // TUPLE1
        case 0x86: MakeTupleN(2); break;                           // TUPLE2
        case 0x87: MakeTupleN(3); break;                           // TUPLE3
        case 0x51: DoPersid(); break;                              // BINPERSID
        case 0x63: DoGlobal(); break;                              // GLOBAL
        case 0x93: DoStackGlobal(); break;                         // STACK_GLOBAL
        case 0x52: DoReduce(); break;                              // REDUCE
        case 0x62: DoBuild(); break;                               // BUILD
        case 0x73: DoSetItem(); break;                             // SETITEM
        case 0x75: DoSetItems(); break;                            // SETITEMS
        case 0x61: DoAppend(); break;                              // APPEND
        case 0x65: DoAppends(); break;                             // APPENDS
        case 0x2e: {                                               // STOP
          if (stack_.empty()) {
            return Err(ErrorCode::ParseError, "pickle: empty stack at STOP");
          }
          return stack_.back();
        }
        default:
          return Err(ErrorCode::Unsupported, fmt::format("pickle: unsupported opcode 0x{:02x}", op));
      }
      if (err_) {
        return Err(code_, msg_);
      }
    }
  }

 private:
  std::span<const std::byte> buf_;
  std::size_t pos_;
  std::vector<Value> stack_;
  std::unordered_map<std::uint32_t, Value> memo_;
  bool err_{false};
  ErrorCode code_{ErrorCode::ParseError};
  std::string msg_;

  void Fail(ErrorCode c, std::string m) {
    if (!err_) {
      err_ = true;
      code_ = c;
      msg_ = std::move(m);
    }
  }

  static Value MakeKind(Value::Kind k) {
    Value v;
    v.kind = k;
    return v;
  }
  static Value MakeInt(std::int64_t n) {
    Value v;
    v.kind = Value::Kind::Int;
    v.i = n;
    return v;
  }
  static Value MakeBool(bool x) {
    Value v;
    v.kind = Value::Kind::Bool;
    v.b = x;
    return v;
  }
  static Value MakeStr(std::string s) {
    Value v;
    v.kind = Value::Kind::Str;
    v.s = std::move(s);
    return v;
  }

  std::uint8_t U8() {
    if (pos_ >= buf_.size()) {
      Fail(ErrorCode::ParseError, "pickle: truncated");
      return 0;
    }
    return std::to_integer<std::uint8_t>(buf_[pos_++]);
  }
  std::uint16_t U16() {
    const std::uint16_t a = U8();
    const std::uint16_t b = U8();
    return static_cast<std::uint16_t>(a | static_cast<std::uint16_t>(b << 8));
  }
  std::uint32_t U32() {
    std::uint32_t v = 0;
    for (std::size_t k = 0; k < 4; ++k) {
      v |= static_cast<std::uint32_t>(U8()) << (8 * k);
    }
    return v;
  }
  std::int32_t I32() { return static_cast<std::int32_t>(U32()); }

  std::int64_t LongN(std::size_t n) {
    if (pos_ + n > buf_.size()) {
      Fail(ErrorCode::ParseError, "pickle: truncated long");
      return 0;
    }
    if (n == 0) {
      return 0;
    }
    if (n > 8) {  // only the discarded magic-number archive has a >8-byte int
      pos_ += n;
      return 0;
    }
    std::uint64_t u = 0;
    for (std::size_t k = 0; k < n; ++k) {
      u |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(buf_[pos_++])) << (8 * k);
    }
    if (n < 8 && (u & (std::uint64_t{1} << (8 * n - 1))) != 0) {  // sign-extend
      u |= ~((std::uint64_t{1} << (8 * n)) - 1);
    }
    return static_cast<std::int64_t>(u);
  }

  std::string Bytes(std::size_t n) {
    if (pos_ + n > buf_.size()) {
      Fail(ErrorCode::ParseError, "pickle: truncated string");
      return {};
    }
    std::string out(reinterpret_cast<const char*>(buf_.data()) + pos_, n);
    pos_ += n;
    return out;
  }
  std::string Line() {
    std::string out;
    while (pos_ < buf_.size()) {
      const char c = static_cast<char>(std::to_integer<std::uint8_t>(buf_[pos_++]));
      if (c == '\n') {
        return out;
      }
      out.push_back(c);
    }
    Fail(ErrorCode::ParseError, "pickle: unterminated line");
    return out;
  }

  void Push(Value v) { stack_.push_back(std::move(v)); }
  Value Pop() {
    if (stack_.empty()) {
      Fail(ErrorCode::ParseError, "pickle: stack underflow");
      return {};
    }
    Value v = std::move(stack_.back());
    stack_.pop_back();
    return v;
  }
  std::vector<Value> PopToMark() {
    std::vector<Value> out;
    while (!stack_.empty() && stack_.back().kind != Value::Kind::Mark) {
      out.push_back(std::move(stack_.back()));
      stack_.pop_back();
    }
    if (stack_.empty()) {
      Fail(ErrorCode::ParseError, "pickle: no MARK");
      return {};
    }
    stack_.pop_back();  // drop the mark
    std::reverse(out.begin(), out.end());
    return out;
  }

  void MemoPut(std::uint32_t key) {
    if (!stack_.empty()) {
      memo_[key] = stack_.back();
    }
  }
  void MemoGet(std::uint32_t key) {
    const auto it = memo_.find(key);
    if (it == memo_.end()) {
      Fail(ErrorCode::ParseError, "pickle: bad memo get");
      return;
    }
    Push(it->second);
  }

  void MakeTupleFromMark() {
    Value t = MakeKind(Value::Kind::Tuple);
    t.items = PopToMark();
    Push(std::move(t));
  }
  void MakeTupleN(std::size_t n) {
    Value t = MakeKind(Value::Kind::Tuple);
    t.items.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
      t.items[n - 1 - k] = Pop();
    }
    Push(std::move(t));
  }

  void DoGlobal() {
    Value g = MakeKind(Value::Kind::Global);
    g.s = Line();   // module
    g.s2 = Line();  // name
    Push(std::move(g));
  }
  void DoStackGlobal() {
    Value name = Pop();
    Value module = Pop();
    Value g = MakeKind(Value::Kind::Global);
    g.s = module.s;
    g.s2 = name.s;
    Push(std::move(g));
  }

  void DoPersid() {
    Value t = Pop();
    if (t.kind != Value::Kind::Tuple || t.items.size() < 5 ||
        t.items[0].kind != Value::Kind::Str || t.items[0].s != "storage") {
      Fail(ErrorCode::Unsupported, "pickle: unexpected persistent_id");
      return;
    }
    if (t.items[1].kind != Value::Kind::Global) {
      Fail(ErrorCode::Unsupported, "pickle: persid storage type");
      return;
    }
    const auto dt = StorageDType(t.items[1].s2);
    if (!dt) {
      Fail(ErrorCode::Unsupported, fmt::format("pickle: storage type '{}'", t.items[1].s2));
      return;
    }
    Value s = MakeKind(Value::Kind::Storage);
    s.s = (t.items[2].kind == Value::Kind::Str) ? t.items[2].s : std::string{};
    s.dt = *dt;
    s.numel = (t.items[4].kind == Value::Kind::Int) ? t.items[4].i : 0;
    Push(std::move(s));
  }

  void DoReduce() {
    Value args = Pop();
    Value fn = Pop();
    if (err_) {
      return;
    }
    if (fn.kind != Value::Kind::Global) {
      Push(MakeKind(Value::Kind::Opaque));
      return;
    }
    if (fn.s2 == "_rebuild_tensor_v2") {
      if (args.kind != Value::Kind::Tuple || args.items.size() < 4 ||
          args.items[0].kind != Value::Kind::Storage) {
        Fail(ErrorCode::Unsupported, "pickle: bad _rebuild_tensor_v2 args");
        return;
      }
      const Value& st = args.items[0];
      Value t = MakeKind(Value::Kind::Tensor);
      t.s = st.s;
      t.dt = st.dt;
      t.numel = st.numel;
      t.i = (args.items[1].kind == Value::Kind::Int) ? args.items[1].i : 0;
      for (const Value& e : args.items[2].items) {
        t.shape.push_back(e.i);
      }
      for (const Value& e : args.items[3].items) {
        t.stride.push_back(e.i);
      }
      Push(std::move(t));
    } else if (fn.s2 == "_rebuild_parameter") {
      if (args.kind == Value::Kind::Tuple && !args.items.empty() &&
          args.items[0].kind == Value::Kind::Tensor) {
        Push(args.items[0]);
      } else {
        Push(MakeKind(Value::Kind::Opaque));
      }
    } else if (fn.s2 == "OrderedDict") {
      Push(MakeKind(Value::Kind::Dict));
    } else {
      Push(MakeKind(Value::Kind::Opaque));  // unknown reduce -> inert, never called
    }
  }

  void DoBuild() {
    Pop();  // state (backward hooks etc.) — discarded; leave the object as-is
  }

  void DoAppend() {
    Value v = Pop();
    if (!stack_.empty() && stack_.back().kind == Value::Kind::Tuple) {
      stack_.back().items.push_back(std::move(v));
    }
  }
  void DoAppends() {
    std::vector<Value> items = PopToMark();
    if (!stack_.empty() && stack_.back().kind == Value::Kind::Tuple) {
      for (Value& it : items) {
        stack_.back().items.push_back(std::move(it));
      }
    }
  }

  void DoSetItem() {
    Value val = Pop();
    Value key = Pop();
    if (!stack_.empty() && stack_.back().kind == Value::Kind::Dict) {
      stack_.back().entries.emplace_back(std::move(key), std::move(val));
    }
  }
  void DoSetItems() {
    std::vector<Value> kv = PopToMark();
    if (err_ || stack_.empty() || stack_.back().kind != Value::Kind::Dict) {
      Fail(ErrorCode::ParseError, "pickle: SETITEMS without a dict");
      return;
    }
    for (std::size_t k = 0; k + 1 < kv.size(); k += 2) {
      stack_.back().entries.emplace_back(std::move(kv[k]), std::move(kv[k + 1]));
    }
  }
};

// True iff the (size, stride) describe a C-contiguous tensor (size-1 dims ignored,
// matching torch's is_contiguous).
bool IsContiguous(const std::vector<std::int64_t>& size, const std::vector<std::int64_t>& stride) {
  if (size.size() != stride.size()) {
    return false;
  }
  std::int64_t expected = 1;
  for (std::size_t k = size.size(); k-- > 0;) {
    if (size[k] != 1) {
      if (stride[k] != expected) {
        return false;
      }
      expected *= size[k];
    }
  }
  return true;
}

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
  if (!magic) {
    return Err(magic.error().code, magic.error().message);
  }
  Unpickler a2(bytes, a1.pos());
  auto proto = a2.Run();
  if (!proto) {
    return Err(proto.error().code, proto.error().message);
  }
  Unpickler a3(bytes, a2.pos());
  auto sys = a3.Run();
  if (!sys) {
    return Err(sys.error().code, sys.error().message);
  }
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
  if (!main) {
    return Err(main.error().code, main.error().message);
  }
  if (main->kind != Value::Kind::Dict) {
    return Err(ErrorCode::Unsupported, "legacy checkpoint root is not a state_dict");
  }

  Unpickler a5(bytes, a4.pos());
  auto keylist = a5.Run();
  if (!keylist) {
    return Err(keylist.error().code, keylist.error().message);
  }
  std::vector<std::string> keys;
  for (const Value& e : keylist->items) {
    if (e.kind != Value::Kind::Str) {
      return Err(ErrorCode::Unsupported, "legacy storage-key list is not all strings");
    }
    keys.push_back(e.s);
  }

  // Collect (name -> tensor descriptor) and the per-storage dtype.
  std::vector<std::pair<std::string, Value>> tensors;
  std::map<std::string, DType> key_dt;
  for (const auto& [k, v] : main->entries) {
    if (k.kind == Value::Kind::Str && v.kind == Value::Kind::Tensor) {
      tensors.emplace_back(k.s, v);
      key_dt[v.s] = v.dt;
    }
  }

  // Walk the storage section: for each key, an int64 element count then the bytes.
  std::size_t sp = a5.pos();
  std::map<std::string, std::vector<std::byte>> store;
  for (const std::string& key : keys) {
    if (sp + 8 > bytes.size()) {
      return Err(ErrorCode::ParseError, "legacy storage: truncated size");
    }
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
    if (sp + nbytes > bytes.size()) {
      return Err(ErrorCode::ParseError, "legacy storage: truncated data");
    }
    store[key].assign(bytes.begin() + static_cast<std::ptrdiff_t>(sp),
                      bytes.begin() + static_cast<std::ptrdiff_t>(sp + nbytes));
    sp += nbytes;
  }

  StateDict sd;
  for (const auto& [name, t] : tensors) {
    if (!IsContiguous(t.shape, t.stride)) {
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
  if (sd.Empty()) {
    return Err(ErrorCode::Unsupported, "no tensors recovered from legacy checkpoint");
  }
  return sd;
}

Result<StateDict> LoadLegacyPth(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return Err(ErrorCode::Io, fmt::format("cannot open '{}'", path.string()));
  }
  const auto size = f.tellg();
  if (size < 0) {
    return Err(ErrorCode::Io, fmt::format("cannot size '{}'", path.string()));
  }
  std::vector<std::byte> buf(static_cast<std::size_t>(size));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(buf.data()), size);
  if (!f) {
    return Err(ErrorCode::Io, fmt::format("cannot read '{}'", path.string()));
  }
  return LoadLegacyPthBytes(buf);
}

}  // namespace detr::weights
