// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Direct-load regression: every registered model must load the authors' OWN native
// checkpoint (from its original repo, downloaded into models/) with 0 missing /
// 0 unexpected / 0 shape-mismatched — no Python conversion, only the model's
// UpstreamRemapper. Each case is skipped when its models/<file> is absent (the
// downloader fetches it: cmake -P scripts/download_models.cmake -- <model>).

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <filesystem>
#include <string>

#include "detr/models/registry.hpp"
#include "detr/weights/pth.hpp"
#include "detr/weights/torch_bridge.hpp"

namespace detr::models {
namespace {

// Repo-root models/ directory, located relative to this source file at build time.
std::filesystem::path ModelsDir() {
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "models";
}

struct UpstreamCase {
  const char* model;  // registered name
  const char* file;   // original checkpoint filename under models/
};

// Grows as each family's UpstreamRemapper is implemented + its weights are linked.
const UpstreamCase kCases[] = {
    {"deformable-detr", "r50_deformable_detr-checkpoint.pth"},
    {"detr-r50", "detr-r50-e632da11.pth"},
    {"detr-r101", "detr-r101-2c7b67e5.pth"},
    {"detr-r50-dc5", "detr-r50-dc5-f0fb7ef5.pth"},
    {"detr-r101-dc5", "detr-r101-dc5-a2e86def.pth"},
    {"rt-detr-s", "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"},
    {"rt-detr-m", "rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"},
    {"rt-detr-l", "rtdetr_r50vd_6x_coco_from_paddle.pth"},
    {"rt-detr-x", "rtdetr_r101vd_6x_coco_from_paddle.pth"},
    {"rt-detrv2-s", "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"},
    {"rt-detrv2-m", "rtdetrv2_r34vd_120e_coco_ema.pth"},
    {"rt-detrv2-l", "rtdetrv2_r50vd_6x_coco_ema.pth"},
    {"rt-detrv2-x", "rtdetrv2_r101vd_6x_coco_from_paddle.pth"},
    {"dfine-n", "dfine_n_coco.pth"},
    {"dfine-s", "dfine_s_coco.pth"},
    {"dfine-m", "dfine_m_coco.pth"},
    {"dfine-l", "dfine_l_coco.pth"},
    {"dfine-x", "dfine_x_coco.pth"},
    {"dfine-s-obj", "dfine_s_obj2coco.pth"},
    {"dfine-m-obj", "dfine_m_obj2coco.pth"},
    {"dfine-l-obj", "dfine_l_obj2coco_e25.pth"},
    {"dfine-x-obj", "dfine_x_obj2coco.pth"},
    {"deim-n", "deim_dfine_n.pth"},
    {"deim-s", "deim_dfine_s.pth"},
    {"deim-m", "deim_dfine_m.pth"},
    {"deim-l", "deim_dfine_l.pth"},
    {"deim-x", "deim_dfine_x.pth"},
    {"deim-rt-s", "deim_rt_r18.pth"},
    {"deim-rt-m", "deim_rt_r34.pth"},
    {"deim-rt-l", "deim_rt_r50.pth"},
};

class UpstreamLoad : public ::testing::TestWithParam<UpstreamCase> {};

TEST_P(UpstreamLoad, LoadsNativeCheckpointZeroZero) {
  const UpstreamCase c = GetParam();
  const auto path = ModelsDir() / c.file;
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << c.file << " absent (run: cmake -P scripts/download_models.cmake -- "
                 << c.model << ")";
  }
  RegisterBuiltins();
  auto model = Registry::Instance().Build(c.model);
  ASSERT_TRUE(model.has_value()) << model.error().message;

  auto sd = weights::LoadPth(path.string());
  ASSERT_TRUE(sd.has_value()) << sd.error().message;
  auto rep = weights::LoadStateDictInto(**model, *sd, (*model)->UpstreamRemapper(), false);
  ASSERT_TRUE(rep.has_value()) << rep.error().message;

  // num_batches_tracked is a training-only BN counter (unused at inference); some
  // upstream checkpoints (e.g. RT-DETR's paddle-converted weights) omit it. Don't
  // count it as a real load discrepancy.
  auto significant = [](const std::vector<std::string>& keys) {
    std::size_t n = 0;
    for (const auto& k : keys)
      if (k.find("num_batches_tracked") == std::string::npos) ++n;
    return n;
  };
  for (const auto& mk : rep->missing) std::cout << "  MISSING " << mk << "\n";
  for (const auto& uk : rep->unexpected) std::cout << "  UNEXPECTED " << uk << "\n";
  for (const auto& sk : rep->mismatched) std::cout << "  MISMATCH " << sk << "\n";
  std::cout << c.model << ": loaded " << rep->loaded << " missing " << rep->missing.size()
            << " unexpected " << rep->unexpected.size() << " mismatched " << rep->mismatched.size()
            << "\n";
  EXPECT_GT(rep->loaded, 0U);
  EXPECT_EQ(significant(rep->missing), 0U);
  EXPECT_EQ(significant(rep->unexpected), 0U);
  EXPECT_EQ(rep->mismatched.size(), 0U);
}

INSTANTIATE_TEST_SUITE_P(NativeCheckpoints, UpstreamLoad, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<UpstreamCase>& i) {
                           std::string n = i.param.model;
                           for (char& ch : n)
                             if (!std::isalnum(static_cast<unsigned char>(ch))) ch = '_';
                           return n;
                         });

}  // namespace
}  // namespace detr::models
