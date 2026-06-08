// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/eval/evaluator.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "detr/data/loader.hpp"
#include "detr/data/sample.hpp"
#include "detr/infer/postprocess.hpp"
#include "detr/log/log.hpp"

namespace detr::eval {

CocoMetrics EvaluateModel(models::IModel& model, const data::Dataset& dataset,
                          data::Split split, int imgsz, int batch, torch::Device device) {
  torch::NoGradGuard no_grad;
  model.eval();
  const int num_classes = model.Meta().num_classes;

  data::DataLoader loader(dataset, split, imgsz, batch, /*seed=*/0);
  std::vector<EvalImage> eval_images;
  eval_images.reserve(loader.NumSamples());

  const std::size_t batches = loader.NumBatches();
  for (std::size_t b = 0; b < batches; ++b) {
    auto loaded = loader.At(b);
    if (!loaded) {
      detr::log::Get("eval").warn("{}", loaded.error().message);
      continue;
    }
    auto images = loaded->images.to(device);
    auto outputs = model.Forward(images);

    const int bs = static_cast<int>(loaded->sizes.size());
    for (int j = 0; j < bs; ++j) {
      const auto [w, h] = loaded->sizes[static_cast<std::size_t>(j)];
      const float w_scale = static_cast<float>(w > 0 ? w : imgsz);
      const float h_scale = static_cast<float>(h > 0 ? h : imgsz);
      const data::Sample& s = dataset.samples[loaded->sample_indices[static_cast<std::size_t>(j)]];

      EvalImage ei;
      ei.gts.reserve(s.boxes.size());
      for (const auto& box : s.boxes) {
        GtBox g;
        g.category_id = box.class_id;
        g.x = (box.cx - box.w / 2.0F) * w_scale;
        g.y = (box.cy - box.h / 2.0F) * h_scale;
        g.w = box.w * w_scale;
        g.h = box.h * h_scale;
        g.iscrowd = false;
        ei.gts.push_back(g);
      }
      ei.dts = infer::PostprocessImage(outputs, j, static_cast<int>(w_scale),
                                       static_cast<int>(h_scale), num_classes);
      eval_images.push_back(std::move(ei));
    }
  }

  std::vector<int> cats;
  cats.reserve(static_cast<std::size_t>(num_classes));
  for (int c = 0; c < num_classes; ++c) {
    cats.push_back(c);
  }
  return CocoEvaluate(eval_images, cats);
}

}  // namespace detr::eval
