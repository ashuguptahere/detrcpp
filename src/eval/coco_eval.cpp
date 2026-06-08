// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Reimplementation of pycocotools COCOeval (bbox). Structure mirrors the
// original: precompute IoUs per (category, image); evaluateImg per (category,
// area, image) producing per-detection match/ignore flags; accumulate over
// images into precision/recall arrays; summarize into the 12 metrics.

#include "detr/eval/coco_eval.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace detr::eval {

namespace {

constexpr int kT = 10;   // IoU thresholds 0.50..0.95
constexpr int kR = 101;  // recall thresholds 0..1
constexpr int kA = 4;    // area ranges: all, small, medium, large
constexpr int kM = 3;    // maxDets: 1, 10, 100

const std::array<double, kT>& IouThrs() {
  static const std::array<double, kT> v = [] {
    std::array<double, kT> a{};
    for (int i = 0; i < kT; ++i) {
      a[static_cast<std::size_t>(i)] = 0.5 + 0.05 * i;
    }
    return a;
  }();
  return v;
}

const std::array<double, kR>& RecThrs() {
  static const std::array<double, kR> v = [] {
    std::array<double, kR> a{};
    for (int i = 0; i < kR; ++i) {
      a[static_cast<std::size_t>(i)] = 0.01 * i;
    }
    return a;
  }();
  return v;
}

const std::array<std::array<double, 2>, kA>& AreaRng() {
  static const std::array<std::array<double, 2>, kA> v = {{
      {0.0, 1e10},        // all
      {0.0, 32.0 * 32},   // small
      {32.0 * 32, 96.0 * 96},
      {96.0 * 96, 1e10},  // large
  }};
  return v;
}

const std::array<int, kM>& MaxDets() {
  static const std::array<int, kM> v = {1, 10, 100};
  return v;
}

double IouDtGt(const DtBox& d, const GtBox& g) {
  const double dx = static_cast<double>(d.x);
  const double dy = static_cast<double>(d.y);
  const double dw = static_cast<double>(d.w);
  const double dh = static_cast<double>(d.h);
  const double gx = static_cast<double>(g.x);
  const double gy = static_cast<double>(g.y);
  const double gw = static_cast<double>(g.w);
  const double gh = static_cast<double>(g.h);
  const double iw = std::min(dx + dw, gx + gw) - std::max(dx, gx);
  const double ih = std::min(dy + dh, gy + gh) - std::max(dy, gy);
  if (iw <= 0.0 || ih <= 0.0) {
    return 0.0;
  }
  const double inter = iw * ih;
  const double dt_area = dw * dh;
  const double gt_area = gw * gh;
  const double denom = g.iscrowd ? dt_area : (dt_area + gt_area - inter);
  return denom <= 0.0 ? 0.0 : inter / denom;
}

// Per-image evaluation output for one (category, area).
struct ImgEval {
  bool valid{false};
  int d{0};
  std::vector<double> dt_scores;  // [d]
  std::vector<char> dt_matched;   // [T*d]
  std::vector<char> dt_ignore;    // [T*d]
  int npig{0};                    // non-ignored ground-truth count
};

// gts: this cat/img ground truths; dts: detections sorted by score desc;
// iou: [D*G] IoUs aligned (dts x gts, original gt order).
ImgEval EvaluateImg(const std::vector<GtBox>& gts, const std::vector<DtBox>& dts,
                    const std::vector<double>& iou, const std::array<double, 2>& area,
                    int max_det) {
  ImgEval e;
  const int g_count = static_cast<int>(gts.size());
  const int d_full = static_cast<int>(dts.size());
  if (g_count == 0 && d_full == 0) {
    return e;  // invalid -> skipped by accumulate
  }
  e.valid = true;

  // Ground-truth ignore flags, then reorder gts so non-ignored come first.
  std::vector<char> gt_ig(static_cast<std::size_t>(g_count));
  for (int g = 0; g < g_count; ++g) {
    const auto& gt = gts[static_cast<std::size_t>(g)];
    const double a = static_cast<double>(gt.w) * static_cast<double>(gt.h);
    gt_ig[static_cast<std::size_t>(g)] =
        static_cast<char>(gt.iscrowd || a < area[0] || a > area[1]);
  }
  std::vector<int> order(static_cast<std::size_t>(g_count));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
    return gt_ig[static_cast<std::size_t>(x)] < gt_ig[static_cast<std::size_t>(y)];
  });

  std::vector<char> gt_ig_r(static_cast<std::size_t>(g_count));
  std::vector<char> gt_crowd_r(static_cast<std::size_t>(g_count));
  for (int gi = 0; gi < g_count; ++gi) {
    const int g = order[static_cast<std::size_t>(gi)];
    gt_ig_r[static_cast<std::size_t>(gi)] = gt_ig[static_cast<std::size_t>(g)];
    gt_crowd_r[static_cast<std::size_t>(gi)] =
        static_cast<char>(gts[static_cast<std::size_t>(g)].iscrowd);
  }

  const int d = std::min(d_full, max_det);
  e.d = d;
  e.dt_scores.resize(static_cast<std::size_t>(d));
  e.dt_matched.assign(static_cast<std::size_t>(kT * d), 0);
  e.dt_ignore.assign(static_cast<std::size_t>(kT * d), 0);
  for (int di = 0; di < d; ++di) {
    e.dt_scores[static_cast<std::size_t>(di)] = dts[static_cast<std::size_t>(di)].score;
  }

  auto iou_r = [&](int di, int gi) {
    return iou[static_cast<std::size_t>(di) * static_cast<std::size_t>(g_count) +
               static_cast<std::size_t>(order[static_cast<std::size_t>(gi)])];
  };

  std::vector<char> gt_matched(static_cast<std::size_t>(kT * g_count), 0);
  for (int t = 0; t < kT; ++t) {
    for (int di = 0; di < d; ++di) {
      double best = std::min(IouThrs()[static_cast<std::size_t>(t)], 1.0 - 1e-10);
      int match = -1;
      for (int gi = 0; gi < g_count; ++gi) {
        if (gt_matched[static_cast<std::size_t>(t * g_count + gi)] != 0 &&
            gt_crowd_r[static_cast<std::size_t>(gi)] == 0) {
          continue;  // gt already taken (crowd gts may match many)
        }
        if (match > -1 && gt_ig_r[static_cast<std::size_t>(match)] == 0 &&
            gt_ig_r[static_cast<std::size_t>(gi)] == 1) {
          break;  // matched a real gt; the rest are ignore-only
        }
        if (iou_r(di, gi) < best) {
          continue;
        }
        best = iou_r(di, gi);
        match = gi;
      }
      if (match == -1) {
        continue;
      }
      e.dt_ignore[static_cast<std::size_t>(t * d + di)] = gt_ig_r[static_cast<std::size_t>(match)];
      e.dt_matched[static_cast<std::size_t>(t * d + di)] = 1;
      gt_matched[static_cast<std::size_t>(t * g_count + match)] = 1;
    }
    // Unmatched detections outside the area range are ignored.
    for (int di = 0; di < d; ++di) {
      if (e.dt_matched[static_cast<std::size_t>(t * d + di)] == 0) {
        const auto& dd = dts[static_cast<std::size_t>(di)];
        const double da = static_cast<double>(dd.w) * static_cast<double>(dd.h);
        if (da < area[0] || da > area[1]) {
          e.dt_ignore[static_cast<std::size_t>(t * d + di)] = 1;
        }
      }
    }
  }

  for (int gi = 0; gi < g_count; ++gi) {
    if (gt_ig_r[static_cast<std::size_t>(gi)] == 0) {
      ++e.npig;
    }
  }
  return e;
}

double MeanValid(const std::vector<double>& vals) {
  double sum = 0;
  int n = 0;
  for (const double v : vals) {
    if (v > -1.0) {
      sum += v;
      ++n;
    }
  }
  return n == 0 ? -1.0 : sum / n;
}

}  // namespace

CocoMetrics CocoEvaluate(const std::vector<EvalImage>& images,
                         const std::vector<int>& category_ids) {
  const int num_cats = static_cast<int>(category_ids.size());
  const int num_imgs = static_cast<int>(images.size());

  auto z = [](int x) { return static_cast<std::size_t>(x); };
  const std::size_t nc = z(num_cats);

  // precision[t][r][k][a][m], recall[t][k][a][m] (flattened), -1 = undefined.
  std::vector<double> precision(z(kT) * z(kR) * nc * z(kA) * z(kM), -1.0);
  std::vector<double> recall(z(kT) * nc * z(kA) * z(kM), -1.0);
  auto pidx = [&](int t, int r, int k, int a, int m) {
    return ((((z(t) * z(kR) + z(r)) * nc + z(k)) * z(kA) + z(a)) * z(kM) + z(m));
  };
  auto ridx = [&](int t, int k, int a, int m) {
    return (((z(t) * nc + z(k)) * z(kA) + z(a)) * z(kM) + z(m));
  };

  for (int k = 0; k < num_cats; ++k) {
    const int cat = category_ids[static_cast<std::size_t>(k)];

    // Per-image: ground truths of this category, detections sorted by score,
    // and the IoU matrix (computed once; area-independent).
    std::vector<std::vector<GtBox>> img_gts(static_cast<std::size_t>(num_imgs));
    std::vector<std::vector<DtBox>> img_dts(static_cast<std::size_t>(num_imgs));
    std::vector<std::vector<double>> img_iou(static_cast<std::size_t>(num_imgs));
    for (int i = 0; i < num_imgs; ++i) {
      const auto& im = images[static_cast<std::size_t>(i)];
      auto& gts = img_gts[static_cast<std::size_t>(i)];
      auto& dts = img_dts[static_cast<std::size_t>(i)];
      for (const auto& g : im.gts) {
        if (g.category_id == cat) {
          gts.push_back(g);
        }
      }
      for (const auto& dd : im.dts) {
        if (dd.category_id == cat) {
          dts.push_back(dd);
        }
      }
      std::stable_sort(dts.begin(), dts.end(),
                       [](const DtBox& x, const DtBox& y) { return x.score > y.score; });
      auto& iou = img_iou[static_cast<std::size_t>(i)];
      iou.resize(dts.size() * gts.size());
      for (std::size_t di = 0; di < dts.size(); ++di) {
        for (std::size_t gi = 0; gi < gts.size(); ++gi) {
          iou[di * gts.size() + gi] = IouDtGt(dts[di], gts[gi]);
        }
      }
    }

    for (int a = 0; a < kA; ++a) {
      // evaluateImg per image with the largest maxDet; accumulate re-caps.
      std::vector<ImgEval> evals(static_cast<std::size_t>(num_imgs));
      for (int i = 0; i < num_imgs; ++i) {
        evals[static_cast<std::size_t>(i)] =
            EvaluateImg(img_gts[static_cast<std::size_t>(i)], img_dts[static_cast<std::size_t>(i)],
                        img_iou[static_cast<std::size_t>(i)], AreaRng()[static_cast<std::size_t>(a)],
                        MaxDets()[kM - 1]);
      }

      for (int m = 0; m < kM; ++m) {
        const int max_det = MaxDets()[static_cast<std::size_t>(m)];

        struct Entry {
          double score;
          std::array<char, kT> matched;
          std::array<char, kT> ignore;
        };
        std::vector<Entry> entries;
        int npig = 0;
        for (int i = 0; i < num_imgs; ++i) {
          const auto& e = evals[static_cast<std::size_t>(i)];
          if (!e.valid) {
            continue;
          }
          npig += e.npig;
          const int take = std::min(e.d, max_det);
          for (int di = 0; di < take; ++di) {
            Entry en;
            en.score = e.dt_scores[static_cast<std::size_t>(di)];
            for (int t = 0; t < kT; ++t) {
              en.matched[static_cast<std::size_t>(t)] =
                  e.dt_matched[static_cast<std::size_t>(t * e.d + di)];
              en.ignore[static_cast<std::size_t>(t)] =
                  e.dt_ignore[static_cast<std::size_t>(t * e.d + di)];
            }
            entries.push_back(en);
          }
        }
        if (npig == 0) {
          continue;  // category has no gt in this area -> leave -1
        }
        std::stable_sort(entries.begin(), entries.end(),
                         [](const Entry& x, const Entry& y) { return x.score > y.score; });

        for (int t = 0; t < kT; ++t) {
          std::vector<double> rc;
          std::vector<double> pr;
          int tp = 0;
          int fp = 0;
          for (const auto& en : entries) {
            if (en.ignore[static_cast<std::size_t>(t)] != 0) {
              continue;
            }
            if (en.matched[static_cast<std::size_t>(t)] != 0) {
              ++tp;
            } else {
              ++fp;
            }
            rc.push_back(static_cast<double>(tp) / npig);
            pr.push_back(static_cast<double>(tp) / (tp + fp));
          }
          recall[ridx(t, k, a, m)] = rc.empty() ? 0.0 : rc.back();

          // Precision envelope (monotonic non-increasing from the right).
          for (int i = static_cast<int>(pr.size()) - 2; i >= 0; --i) {
            pr[static_cast<std::size_t>(i)] =
                std::max(pr[static_cast<std::size_t>(i)], pr[static_cast<std::size_t>(i + 1)]);
          }
          // Interpolate to the 101 recall thresholds.
          std::size_t j = 0;
          for (int r = 0; r < kR; ++r) {
            while (j < rc.size() && rc[j] < RecThrs()[static_cast<std::size_t>(r)]) {
              ++j;
            }
            precision[pidx(t, r, k, a, m)] = (j < pr.size()) ? pr[j] : 0.0;
          }
        }
      }
    }
  }

  // Summarize. AP averages precision over T,R,cats (valid); AR averages recall.
  auto ap = [&](int t_lo, int t_hi, int a, int m) {
    std::vector<double> vals;
    for (int t = t_lo; t < t_hi; ++t) {
      for (int r = 0; r < kR; ++r) {
        for (int k = 0; k < num_cats; ++k) {
          vals.push_back(precision[pidx(t, r, k, a, m)]);
        }
      }
    }
    return MeanValid(vals);
  };
  auto ar = [&](int a, int m) {
    std::vector<double> vals;
    for (int t = 0; t < kT; ++t) {
      for (int k = 0; k < num_cats; ++k) {
        vals.push_back(recall[ridx(t, k, a, m)]);
      }
    }
    return MeanValid(vals);
  };

  constexpr int kAll = 0;
  constexpr int kSmall = 1;
  constexpr int kMedium = 2;
  constexpr int kLarge = 3;
  constexpr int kMax100 = 2;
  constexpr int kT50 = 0;
  constexpr int kT75 = 5;

  CocoMetrics out;
  out.ap = ap(0, kT, kAll, kMax100);
  out.ap50 = ap(kT50, kT50 + 1, kAll, kMax100);
  out.ap75 = ap(kT75, kT75 + 1, kAll, kMax100);
  out.ap_small = ap(0, kT, kSmall, kMax100);
  out.ap_medium = ap(0, kT, kMedium, kMax100);
  out.ap_large = ap(0, kT, kLarge, kMax100);
  out.ar1 = ar(kAll, 0);
  out.ar10 = ar(kAll, 1);
  out.ar100 = ar(kAll, 2);
  out.ar_small = ar(kSmall, kMax100);
  out.ar_medium = ar(kMedium, kMax100);
  out.ar_large = ar(kLarge, kMax100);
  return out;
}

}  // namespace detr::eval
