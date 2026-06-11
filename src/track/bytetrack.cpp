// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/track/bytetrack.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "detr/core/assignment.hpp"

namespace detr::track {

namespace {

constexpr float kStdPos = 1.0F / 20.0F;   // std_weight_position
constexpr float kStdVel = 1.0F / 160.0F;  // std_weight_velocity
constexpr double kBig = 1e6;              // forbidden-match cost

// xyxy -> measurement (cx, cy, a=w/h, h).
std::array<float, 4> ToMeasure(const Box& b) {
  const float w = b[2] - b[0];
  const float h = b[3] - b[1];
  const float hh = (h > 1e-6F) ? h : 1e-6F;
  return {b[0] + w * 0.5F, b[1] + h * 0.5F, w / hh, h};
}

float IoU(const Box& a, const Box& b) {
  const float x1 = std::max(a[0], b[0]);
  const float y1 = std::max(a[1], b[1]);
  const float x2 = std::min(a[2], b[2]);
  const float y2 = std::min(a[3], b[3]);
  const float iw = std::max(0.0F, x2 - x1);
  const float ih = std::max(0.0F, y2 - y1);
  const float inter = iw * ih;
  const float area_a = std::max(0.0F, a[2] - a[0]) * std::max(0.0F, a[3] - a[1]);
  const float area_b = std::max(0.0F, b[2] - b[0]) * std::max(0.0F, b[3] - b[1]);
  const float uni = area_a + area_b - inter;
  return uni > 0.0F ? inter / uni : 0.0F;
}

// Cholesky factor of a 4x4 SPD matrix S (row-major) into lower L (row-major).
bool Chol4(const std::array<float, 16>& s, std::array<float, 16>& l) {
  l.fill(0.0F);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      float sum = s[i * 4 + j];
      for (std::size_t k = 0; k < j; ++k) {
        sum -= l[i * 4 + k] * l[j * 4 + k];
      }
      if (i == j) {
        if (sum <= 0.0F) {
          return false;
        }
        l[i * 4 + j] = std::sqrt(sum);
      } else {
        l[i * 4 + j] = sum / l[j * 4 + j];
      }
    }
  }
  return true;
}

// Solve S x = b given the Cholesky factor L (S = L Lᵀ).
std::array<float, 4> CholSolve(const std::array<float, 16>& l, const std::array<float, 4>& b) {
  std::array<float, 4> y{};
  std::array<float, 4> x{};
  for (std::size_t i = 0; i < 4; ++i) {
    float sum = b[i];
    for (std::size_t k = 0; k < i; ++k) {
      sum -= l[i * 4 + k] * y[k];
    }
    y[i] = sum / l[i * 4 + i];
  }
  for (std::size_t ii = 4; ii-- > 0;) {
    float sum = y[ii];
    for (std::size_t k = ii + 1; k < 4; ++k) {
      sum -= l[k * 4 + ii] * x[k];
    }
    x[ii] = sum / l[ii * 4 + ii];
  }
  return x;
}

}  // namespace

KalmanBoxTracker::KalmanBoxTracker(const Box& xyxy, float sc, int lbl, int track_id)
    : id(track_id), label(lbl), score(sc), state(TrackState::kTracked), hits(1) {
  const auto z = ToMeasure(xyxy);
  for (std::size_t i = 0; i < 4; ++i) {
    mean_[i] = z[i];
    mean_[i + 4] = 0.0F;
  }
  const float h = z[3];
  const std::array<float, 8> std{2 * kStdPos * h, 2 * kStdPos * h, 1e-2F,        2 * kStdPos * h,
                                 10 * kStdVel * h, 10 * kStdVel * h, 1e-5F, 10 * kStdVel * h};
  cov_.fill(0.0F);
  for (std::size_t i = 0; i < 8; ++i) {
    cov_[i * 8 + i] = std[i] * std[i];
  }
}

void KalmanBoxTracker::Predict() {
  // Constant-velocity transition: cx += vcx, ... (F adds the velocity block).
  for (std::size_t i = 0; i < 4; ++i) {
    mean_[i] += mean_[i + 4];
  }
  // cov = F cov Fᵀ + Q. F adds row/col i+4 into i for i<4.
  std::array<float, 64> c = cov_;
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      c[i * 8 + j] += cov_[(i + 4) * 8 + j];  // left-multiply by F
    }
  }
  std::array<float, 64> fcf = c;
  for (std::size_t j = 0; j < 4; ++j) {
    for (std::size_t i = 0; i < 8; ++i) {
      fcf[i * 8 + j] += c[i * 8 + (j + 4)];  // right-multiply by Fᵀ
    }
  }
  const float h = mean_[3];
  const std::array<float, 8> q{kStdPos * h, kStdPos * h, 1e-2F, kStdPos * h,
                               kStdVel * h, kStdVel * h, 1e-5F, kStdVel * h};
  for (std::size_t i = 0; i < 8; ++i) {
    fcf[i * 8 + i] += q[i] * q[i];
  }
  cov_ = fcf;
  ++age;
  ++time_since_update;
}

void KalmanBoxTracker::Update(const Box& xyxy, float sc) {
  const auto z = ToMeasure(xyxy);
  const float h = mean_[3];
  // S = H cov Hᵀ + R = top-left 4x4 of cov + diag(R²).
  const std::array<float, 4> r{kStdPos * h, kStdPos * h, 1e-1F, kStdPos * h};
  std::array<float, 16> s{};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      s[i * 4 + j] = cov_[i * 8 + j];
    }
    s[i * 4 + i] += r[i] * r[i];
  }
  std::array<float, 16> l{};
  if (!Chol4(s, l)) {
    return;  // numerically non-PD; skip the correction (keep the prediction)
  }
  // K = cov Hᵀ S⁻¹ ; K row i = solve(S, cov[i][0..3]).
  std::array<float, 32> k{};  // 8x4
  for (std::size_t i = 0; i < 8; ++i) {
    const std::array<float, 4> mi{cov_[i * 8 + 0], cov_[i * 8 + 1], cov_[i * 8 + 2],
                                  cov_[i * 8 + 3]};
    const auto ki = CholSolve(l, mi);
    for (std::size_t j = 0; j < 4; ++j) {
      k[i * 4 + j] = ki[j];
    }
  }
  // innovation y = z - H mean ; mean += K y.
  std::array<float, 4> y{};
  for (std::size_t j = 0; j < 4; ++j) {
    y[j] = z[j] - mean_[j];
  }
  for (std::size_t i = 0; i < 8; ++i) {
    float d = 0.0F;
    for (std::size_t j = 0; j < 4; ++j) {
      d += k[i * 4 + j] * y[j];
    }
    mean_[i] += d;
  }
  // cov -= K (H cov) ; H cov = first 4 rows of cov.
  std::array<float, 64> nc = cov_;
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t col = 0; col < 8; ++col) {
      float d = 0.0F;
      for (std::size_t j = 0; j < 4; ++j) {
        d += k[i * 4 + j] * cov_[j * 8 + col];
      }
      nc[i * 8 + col] -= d;
    }
  }
  cov_ = nc;

  score = sc;
  ++hits;
  time_since_update = 0;
  state = TrackState::kTracked;
}

Box KalmanBoxTracker::AsXyxy() const {
  const float w = mean_[2] * mean_[3];
  const float h = mean_[3];
  return {mean_[0] - w * 0.5F, mean_[1] - h * 0.5F, mean_[0] + w * 0.5F, mean_[1] + h * 0.5F};
}

Track KalmanBoxTracker::View() const {
  return Track{id, AsXyxy(), score, label, state, age, hits, time_since_update};
}

ByteTracker::ByteTracker(ByteTrackConfig cfg) : cfg_(cfg) {}

void ByteTracker::Reset() {
  tracks_.clear();
  next_id_ = 1;
  frame_ = 0;
}

namespace {

// One association round: rows = tracks (indices into a global pool via row_map),
// cols = detections (indices via col_map). Matches surviving the IoU gate call
// track.Update; matched tracks/dets are flagged. Returns nothing (mutates flags).
void Associate(std::vector<KalmanBoxTracker>& tracks, const std::vector<std::size_t>& row_map,
               const std::vector<Box>& boxes, const std::vector<float>& scores,
               const std::vector<int>& labels, const std::vector<std::size_t>& col_map,
               float gate, bool per_class, std::vector<char>& track_matched,
               std::vector<char>& det_matched) {
  const std::size_t nr = row_map.size();
  const std::size_t nc = col_map.size();
  if (nr == 0 || nc == 0) {
    return;
  }
  std::vector<double> cost(nr * nc);
  for (std::size_t r = 0; r < nr; ++r) {
    const Box pred = tracks[row_map[r]].AsXyxy();
    for (std::size_t c = 0; c < nc; ++c) {
      const std::size_t dc = col_map[c];
      const bool blocked = per_class && tracks[row_map[r]].label != labels[dc];
      cost[r * nc + c] = blocked ? kBig : 1.0 - static_cast<double>(IoU(pred, boxes[dc]));
    }
  }
  const auto pairs = core::LinearSumAssignment(cost, static_cast<int>(nr), static_cast<int>(nc));
  for (const auto& [r, c] : pairs) {
    const std::size_t rr = static_cast<std::size_t>(r);
    const std::size_t cc = static_cast<std::size_t>(c);
    if (cost[rr * nc + cc] <= static_cast<double>(gate)) {  // cost=1-IoU, gate=match_thresh
      auto& trk = tracks[row_map[rr]];
      const std::size_t dc = col_map[cc];
      trk.Update(boxes[dc], scores[dc]);
      trk.label = labels[dc];
      track_matched[row_map[rr]] = 1;
      det_matched[dc] = 1;
    }
  }
}

}  // namespace

std::vector<Track> ByteTracker::Update(const std::vector<Box>& boxes,
                                       const std::vector<float>& scores,
                                       const std::vector<int>& labels) {
  ++frame_;
  for (auto& t : tracks_) {
    t.Predict();
  }

  // Split detections by score band.
  std::vector<std::size_t> high;
  std::vector<std::size_t> low;
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (scores[i] >= cfg_.track_high_thresh) {
      high.push_back(i);
    } else if (scores[i] >= cfg_.track_low_thresh) {
      low.push_back(i);
    }
  }

  std::vector<char> track_matched(tracks_.size(), 0);
  std::vector<char> det_matched(boxes.size(), 0);

  // Stage 1: high-score dets vs all tracks.
  std::vector<std::size_t> all_rows(tracks_.size());
  for (std::size_t i = 0; i < tracks_.size(); ++i) {
    all_rows[i] = i;
  }
  Associate(tracks_, all_rows, boxes, scores, labels, high, cfg_.match_thresh, cfg_.per_class,
            track_matched, det_matched);

  // Stage 2: low-score dets vs tracks still unmatched.
  std::vector<std::size_t> rem_rows;
  for (std::size_t i = 0; i < tracks_.size(); ++i) {
    if (track_matched[i] == 0) {
      rem_rows.push_back(i);
    }
  }
  Associate(tracks_, rem_rows, boxes, scores, labels, low, cfg_.match_thresh_low, cfg_.per_class,
            track_matched, det_matched);

  // Mark expired coasting tracks for removal.
  for (auto& t : tracks_) {
    if (t.time_since_update > cfg_.track_buffer) {
      t.state = TrackState::kRemoved;
    } else if (t.time_since_update > 0) {
      t.state = TrackState::kLost;
    }
  }

  // Spawn new tracks from unmatched high-score detections above the spawn gate.
  for (const std::size_t d : high) {
    if (det_matched[d] == 0 && scores[d] >= cfg_.new_track_thresh) {
      tracks_.emplace_back(boxes[d], scores[d], labels[d], next_id_++);
    }
  }

  // Reap removed tracks.
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [](const KalmanBoxTracker& t) {
                                 return t.state == TrackState::kRemoved;
                               }),
                tracks_.end());

  // Return the tracks updated (or spawned) this frame.
  std::vector<Track> out;
  for (const auto& t : tracks_) {
    if (t.time_since_update == 0) {
      out.push_back(t.View());
    }
  }
  return out;
}

}  // namespace detr::track
