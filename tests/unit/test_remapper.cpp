// Copyright 2026 detrcpp authors. Apache-2.0.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "detr/weights/remapper.hpp"
#include "detr/weights/state_dict.hpp"
#include "detr/weights/tensor.hpp"

namespace detr::weights {
namespace {

RawTensor Scalar() {
  RawTensor t;
  t.dtype = DType::F32;
  t.shape = {1};
  t.data.resize(sizeof(float));
  return t;
}

TEST(Remapper, StripPrefix) {
  WeightRemapper r;
  r.StripPrefix("model.");
  EXPECT_EQ(r.Map("model.backbone.conv.weight").value(), "backbone.conv.weight");
  EXPECT_EQ(r.Map("class_embed.weight").value(), "class_embed.weight");  // no prefix, unchanged
}

TEST(Remapper, RenameAndRegex) {
  WeightRemapper r;
  r.Rename("transformer.encoder.norm.weight", "encoder.final_norm.weight")
      .ReplaceRegex("self_attn", "attention");
  EXPECT_EQ(r.Map("transformer.encoder.norm.weight").value(), "encoder.final_norm.weight");
  EXPECT_EQ(r.Map("layers.0.self_attn.in_proj_weight").value(),
            "layers.0.attention.in_proj_weight");
}

TEST(Remapper, DropMatchingKeys) {
  WeightRemapper r;
  r.Drop("num_batches_tracked");
  EXPECT_FALSE(r.Map("backbone.bn1.num_batches_tracked").has_value());
  EXPECT_TRUE(r.Map("backbone.bn1.running_mean").has_value());
}

TEST(Remapper, ApplyPreservesDataAndOrderAndMeta) {
  StateDict src;
  src.Set("model.a.weight", Scalar());
  src.Set("model.bn.num_batches_tracked", Scalar());
  src.Set("model.b.bias", Scalar());
  src.SetMeta("format", "pt");

  WeightRemapper r;
  r.StripPrefix("model.").Drop("num_batches_tracked");
  StateDict out = r.Apply(src);

  ASSERT_EQ(out.Size(), 2U);
  EXPECT_EQ(out.Keys()[0], "a.weight");
  EXPECT_EQ(out.Keys()[1], "b.bias");
  EXPECT_FALSE(out.Contains("bn.num_batches_tracked"));
  EXPECT_EQ(out.GetMeta("format").value_or(""), "pt");
}

// A [rows, 1] F32 tensor whose element i is float(i).
RawTensor Rows(std::int64_t rows) {
  RawTensor t;
  t.dtype = DType::F32;
  t.shape = {rows, 1};
  t.data.resize(static_cast<std::size_t>(rows) * sizeof(float));
  for (std::int64_t i = 0; i < rows; ++i) {
    const float v = static_cast<float>(i);
    std::memcpy(t.data.data() + static_cast<std::size_t>(i) * sizeof(float), &v, sizeof(float));
  }
  return t;
}

float RowAt(const RawTensor& t, std::int64_t i) {
  float v = 0;
  std::memcpy(&v, t.data.data() + static_cast<std::size_t>(i) * sizeof(float), sizeof(float));
  return v;
}

TEST(Remapper, SplitRowsFusedQkv) {
  StateDict src;
  src.Set("blocks.0.attn.qkv.weight", Rows(6));  // [6,1]: q=0,1 k=2,3 v=4,5
  WeightRemapper r;
  r.SplitRows("^(blocks\\.[0-9]+)\\.attn\\.qkv\\.weight$",
              {"$1.q.weight", "$1.k.weight", "$1.v.weight"});
  StateDict out = r.Apply(src);
  ASSERT_EQ(out.Size(), 3U);
  const RawTensor* q = out.Find("blocks.0.q.weight");
  const RawTensor* k = out.Find("blocks.0.k.weight");
  const RawTensor* v = out.Find("blocks.0.v.weight");
  ASSERT_NE(q, nullptr);
  ASSERT_NE(k, nullptr);
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(q->shape, (std::vector<std::int64_t>{2, 1}));
  EXPECT_FLOAT_EQ(RowAt(*q, 0), 0.0F);
  EXPECT_FLOAT_EQ(RowAt(*k, 0), 2.0F);
  EXPECT_FLOAT_EQ(RowAt(*v, 1), 5.0F);
}

TEST(Remapper, SliceRowsKeepsLeadingRows) {
  StateDict src;
  src.Set("query_feat.weight", Rows(8));
  WeightRemapper r;
  r.ReplaceRegex("^query_feat\\.weight$", "query_feat").SliceRows("^query_feat$", 3);
  StateDict out = r.Apply(src);
  const RawTensor* q = out.Find("query_feat");
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->shape, (std::vector<std::int64_t>{3, 1}));
  EXPECT_FLOAT_EQ(RowAt(*q, 2), 2.0F);
}

}  // namespace
}  // namespace detr::weights
