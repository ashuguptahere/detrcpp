// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Legacy (pre-1.6) .pth unpickler. We cannot call torch.save (no Python), so a
// small in-test writer emits the exact 5-archive legacy layout the reader expects
// and we round-trip through it; plus adversarial cases (truncation, unknown
// global, big-endian) assert the safe failure modes.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "detr/weights/legacy_pickle.hpp"
#include "detr/weights/tensor.hpp"

namespace detr::weights {
namespace {

// --- minimal legacy-pickle writer (emits only the opcodes the reader needs) ---

struct W {
  std::vector<std::byte> b;
  void U8(unsigned v) { b.push_back(static_cast<std::byte>(v & 0xFF)); }
  void U16(unsigned v) {
    U8(v);
    U8(v >> 8);
  }
  void U32(std::uint32_t v) {
    for (int k = 0; k < 4; ++k) {
      U8(static_cast<unsigned>(v >> (8 * k)));
    }
  }
  void I64(std::int64_t v) {
    auto u = static_cast<std::uint64_t>(v);
    for (int k = 0; k < 8; ++k) {
      U8(static_cast<unsigned>(u >> (8 * k)));
    }
  }
  void Raw(const std::vector<std::byte>& d) { b.insert(b.end(), d.begin(), d.end()); }
  void Proto() {
    U8(0x80);
    U8(2);
  }
  void Stop() { U8('.'); }
  void Int(std::int64_t n) {
    if (n >= 0 && n < 256) {
      U8('K');
      U8(static_cast<unsigned>(n));
    } else if (n >= 0 && n < 65536) {
      U8('M');
      U16(static_cast<unsigned>(n));
    } else {
      U8('J');
      U32(static_cast<std::uint32_t>(n));
    }
  }
  void Str(const std::string& s) {
    U8('X');
    U32(static_cast<std::uint32_t>(s.size()));
    for (char c : s) {
      U8(static_cast<unsigned char>(c));
    }
  }
  void Global(const std::string& mod, const std::string& name) {
    U8('c');
    for (char c : mod) {
      U8(static_cast<unsigned char>(c));
    }
    U8('\n');
    for (char c : name) {
      U8(static_cast<unsigned char>(c));
    }
    U8('\n');
  }
};

struct Spec {
  std::string name;
  std::string key;
  DType dtype;
  std::string storage_class;
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;
};

std::vector<std::int64_t> NaturalStride(const std::vector<std::int64_t>& shape) {
  std::vector<std::int64_t> st(shape.size());
  std::int64_t acc = 1;
  for (std::size_t k = shape.size(); k-- > 0;) {
    st[k] = acc;
    acc *= shape[k];
  }
  return st;
}

std::vector<std::byte> WriteLegacy(const std::vector<Spec>& specs, bool little_endian = true,
                                   const std::string& rebuild = "_rebuild_tensor_v2") {
  W w;
  // #1 magic (a LONG1, value irrelevant — discarded by the reader)
  w.Proto();
  w.U8(0x8a);
  w.U8(2);
  w.U8(0x34);
  w.U8(0x12);
  w.Stop();
  // #2 protocol version
  w.Proto();
  w.Int(1001);
  w.Stop();
  // #3 sys_info: { little_endian: <bool> }
  w.Proto();
  w.U8('}');
  w.Str("little_endian");
  w.U8(little_endian ? 0x88 : 0x89);  // NEWTRUE / NEWFALSE
  w.U8('s');                          // SETITEM
  w.Stop();
  // #4 main state_dict (OrderedDict[str -> _rebuild_tensor_v2(...)])
  w.Proto();
  w.Global("collections", "OrderedDict");
  w.U8(')');  // EMPTY_TUPLE
  w.U8('R');  // REDUCE -> empty odict
  w.U8('(');  // MARK (items)
  for (const Spec& s : specs) {
    w.Str(s.name);
    w.Global("torch._utils", rebuild);
    w.U8('(');  // MARK (args)
    // storage persid tuple: ('storage', <StorageClass>, key, 'cpu', numel)
    w.U8('(');  // MARK
    w.Str("storage");
    w.Global("torch", s.storage_class);
    w.Str(s.key);
    w.Str("cpu");
    const auto numel = static_cast<std::int64_t>(s.data.size() / DTypeSize(s.dtype));
    w.Int(numel);
    w.U8('t');  // TUPLE
    w.U8('Q');  // BINPERSID
    w.Int(0);   // storage_offset
    w.U8('(');  // size tuple
    for (std::int64_t d : s.shape) {
      w.Int(d);
    }
    w.U8('t');
    w.U8('(');  // stride tuple
    for (std::int64_t d : NaturalStride(s.shape)) {
      w.Int(d);
    }
    w.U8('t');
    w.U8(0x89);  // requires_grad = False
    w.U8('N');   // backward_hooks = None
    w.U8('t');   // TUPLE (args)
    w.U8('R');   // REDUCE -> Tensor
  }
  w.U8('u');  // SETITEMS
  w.Stop();
  // #5 storage-key list
  w.Proto();
  w.U8(']');  // EMPTY_LIST
  w.U8('(');  // MARK
  for (const Spec& s : specs) {
    w.Str(s.key);
  }
  w.U8('e');  // APPENDS
  w.Stop();
  // storage payloads, in key order: int64 numel then raw bytes
  for (const Spec& s : specs) {
    w.I64(static_cast<std::int64_t>(s.data.size() / DTypeSize(s.dtype)));
    w.Raw(s.data);
  }
  return w.b;
}

std::vector<std::byte> F32Bytes(const std::vector<float>& v) {
  std::vector<std::byte> out(v.size() * 4);
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}
std::vector<std::byte> I64Bytes(const std::vector<std::int64_t>& v) {
  std::vector<std::byte> out(v.size() * 8);
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

TEST(LegacyPickle, RoundTripsTwoTensors) {
  std::vector<Spec> specs{
      {"backbone.weight", "0", DType::F32, "FloatStorage", {2, 2},
       F32Bytes({1.5F, -2.0F, 3.25F, 4.0F})},
      {"head.bias", "1", DType::I64, "LongStorage", {3}, I64Bytes({7, 8, 9})}};
  auto bytes = WriteLegacy(specs);

  auto sd = LoadLegacyPthBytes(bytes);
  ASSERT_TRUE(sd.has_value()) << sd.error().message;
  EXPECT_EQ(sd->Size(), 2U);

  const RawTensor* w = sd->Find("backbone.weight");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->dtype, DType::F32);
  EXPECT_EQ(w->shape, (std::vector<std::int64_t>{2, 2}));
  std::vector<float> wv(4);
  std::memcpy(wv.data(), w->data.data(), w->data.size());
  EXPECT_FLOAT_EQ(wv[0], 1.5F);
  EXPECT_FLOAT_EQ(wv[3], 4.0F);

  const RawTensor* h = sd->Find("head.bias");
  ASSERT_NE(h, nullptr);
  EXPECT_EQ(h->dtype, DType::I64);
  EXPECT_EQ(h->shape, (std::vector<std::int64_t>{3}));
  std::vector<std::int64_t> hv(3);
  std::memcpy(hv.data(), h->data.data(), h->data.size());
  EXPECT_EQ(hv, (std::vector<std::int64_t>{7, 8, 9}));
}

TEST(LegacyPickle, TruncatedStreamErrors) {
  std::vector<Spec> specs{{"w", "0", DType::F32, "FloatStorage", {2}, F32Bytes({1.0F, 2.0F})}};
  auto bytes = WriteLegacy(specs);
  bytes.resize(bytes.size() / 2);  // cut it in half
  auto sd = LoadLegacyPthBytes(bytes);
  EXPECT_FALSE(sd.has_value());
}

TEST(LegacyPickle, UnknownGlobalYieldsNoTensors) {
  // An unrecognized reduce becomes inert (never dispatched) -> no tensors -> a
  // clean Unsupported, never a wrong tensor.
  std::vector<Spec> specs{{"w", "0", DType::F32, "FloatStorage", {2}, F32Bytes({1.0F, 2.0F})}};
  auto bytes = WriteLegacy(specs, /*little_endian=*/true, /*rebuild=*/"_evil_call");
  auto sd = LoadLegacyPthBytes(bytes);
  ASSERT_FALSE(sd.has_value());
  EXPECT_EQ(sd.error().code, core::ErrorCode::Unsupported);
}

TEST(LegacyPickle, BigEndianRejected) {
  std::vector<Spec> specs{{"w", "0", DType::F32, "FloatStorage", {2}, F32Bytes({1.0F, 2.0F})}};
  auto bytes = WriteLegacy(specs, /*little_endian=*/false);
  auto sd = LoadLegacyPthBytes(bytes);
  ASSERT_FALSE(sd.has_value());
  EXPECT_EQ(sd.error().code, core::ErrorCode::Unsupported);
}

// Opt-in check against a real legacy .pth (CI-skipped unless DETR_LEGACY_PTH is set
// to such a file). Prints a few tensors for manual inspection.
TEST(LegacyPickle, RealFileIfPresent) {
  const char* env = std::getenv("DETR_LEGACY_PTH");
  if (env == nullptr) {
    GTEST_SKIP() << "set DETR_LEGACY_PTH to a real legacy (pre-1.6) .pth to run this";
  }
  const std::filesystem::path path(env);
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "DETR_LEGACY_PTH not found: " << env;
  }
  auto sd = LoadLegacyPth(path);
  ASSERT_TRUE(sd.has_value()) << sd.error().message;
  EXPECT_GT(sd->Size(), 0U);
  std::cout << "loaded " << sd->Size() << " tensors, " << sd->TotalBytes() << " bytes\n";
  std::size_t shown = 0;
  for (const auto& name : sd->Keys()) {
    const RawTensor* t = sd->Find(name);
    std::cout << "  " << name << " " << DTypeName(t->dtype) << " [";
    for (const std::int64_t d : t->shape) {
      std::cout << d << ",";
    }
    std::cout << "]\n";
    if (++shown >= 8) {
      break;
    }
  }
}

}  // namespace
}  // namespace detr::weights
