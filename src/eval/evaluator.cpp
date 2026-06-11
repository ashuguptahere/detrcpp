// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/eval/evaluator.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "detr/data/loader.hpp"
#include "detr/data/sample.hpp"
#include "detr/infer/postprocess.hpp"
#include "detr/infer/preprocess.hpp"
#include "detr/io/image.hpp"
#include "detr/log/log.hpp"
#include "detr/log/timer.hpp"

namespace detr::eval {

namespace {

// Ground truth for one image: normalized cxcywh -> absolute xywh.
EvalImage MakeGt(const data::Sample& s, float w_scale, float h_scale) {
  EvalImage ei;
  ei.gts.reserve(s.boxes.size());
  for (const auto& box : s.boxes) {
    GtBox g;
    g.category_id = box.class_id;
    g.x = (box.cx - box.w / 2.0F) * w_scale;
    g.y = (box.cy - box.h / 2.0F) * h_scale;
    g.w = box.w * w_scale;
    g.h = box.h * h_scale;
    g.iscrowd = box.iscrowd;
    // Normalized COCO area -> absolute pixels in the same space as the box.
    g.area = box.area > 0.0F ? box.area * w_scale * h_scale : g.w * g.h;
    ei.gts.push_back(g);
  }
  return ei;
}

}  // namespace

CocoMetrics EvaluateModel(models::IModel& model, const data::Dataset& dataset, data::Split split,
                          int imgsz, int batch, torch::Device device, int max_images,
                          bool aspect_preserve, int max_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  const detr::log::Stopwatch sw;
  const int num_classes = model.Meta().num_classes;
  const bool focal = model.Meta().focal;
  // RT-DETR (imagenet_norm=false) evals one image at a time with a square [0,1]
  // resize; DETR-family uses aspect-preserving + ImageNet norm when --aspect.
  const bool norm = model.Meta().imagenet_norm;
  std::vector<EvalImage> eval_images;

  if (aspect_preserve || !norm) {
    // Single-image path (no padding/mask needed at batch 1): aspect-preserving for
    // DETR, or square [0,1] for RT-DETR.
    const auto indices = dataset.IndicesOf(split);
    const std::size_t limit = max_images > 0
                                  ? std::min(static_cast<std::size_t>(max_images), indices.size())
                                  : indices.size();
    eval_images.reserve(limit);
    for (std::size_t k = 0; k < limit; ++k) {
      const data::Sample& s = dataset.samples[indices[k]];
      auto rgb = detr::io::LoadRgb(s.image_path);
      if (!rgb) {
        detr::log::Get("eval").warn("{}", rgb.error().message);
        continue;
      }
      auto input = (norm ? infer::PreprocessImageAspect(*rgb, imgsz, max_size)
                         : infer::PreprocessImage(*rgb, imgsz, /*normalize=*/false))
                       .to(device);
      auto outputs = model.Forward(input);
      EvalImage ei = MakeGt(s, static_cast<float>(rgb->width), static_cast<float>(rgb->height));
      ei.dts = infer::PostprocessImage(outputs, 0, rgb->width, rgb->height, num_classes, focal);
      eval_images.push_back(std::move(ei));
      if ((k + 1) % 500 == 0) {
        detr::log::Get("eval").info("evaluated {}/{}", k + 1, limit);
      }
    }
  } else {
    data::DataLoader loader(dataset, split, imgsz, batch, /*seed=*/0);
    eval_images.reserve(loader.NumSamples());
    const std::size_t batches = loader.NumBatches();
    for (std::size_t b = 0; b < batches; ++b) {
      if (max_images > 0 && eval_images.size() >= static_cast<std::size_t>(max_images)) {
        break;
      }
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
        const data::Sample& s =
            dataset.samples[loaded->sample_indices[static_cast<std::size_t>(j)]];
        EvalImage ei = MakeGt(s, w_scale, h_scale);
        ei.dts = infer::PostprocessImage(outputs, j, static_cast<int>(w_scale),
                                         static_cast<int>(h_scale), num_classes, focal);
        eval_images.push_back(std::move(ei));
      }
    }
  }

  const double secs = sw.ElapsedSec();
  const double ips = secs > 0.0 ? static_cast<double>(eval_images.size()) / secs : 0.0;
  detr::log::Get("eval").info("evaluated {} images in {:.1f}s ({:.1f} img/s)", eval_images.size(),
                              secs, ips);

  std::vector<int> cats;
  cats.reserve(static_cast<std::size_t>(num_classes));
  for (int c = 0; c < num_classes; ++c) {
    cats.push_back(c);
  }
  return CocoEvaluate(eval_images, cats);
}

}  // namespace detr::eval
