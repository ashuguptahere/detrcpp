// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/remapper.hpp"

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace detr::weights
