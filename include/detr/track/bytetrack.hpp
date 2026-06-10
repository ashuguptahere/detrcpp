// Copyright 2026 detrcpp authors. Apache-2.0.
//
// ByteTrack-style multi-object tracker. Torch-free: an 8-dim constant-velocity
// Kalman filter per track + two-stage IoU association (high-score then low-score
// detections) reusing core::LinearSumAssignment, with a coast-then-reap lifecycle.
// Boxes are absolute-pixel xyxy. Builds in the lightweight (no-LibTorch) config.

#pragma once

#include <array>
#include <vector>

namespace detr::track {

// Axis-aligned box, absolute pixels, corner form {x1, y1, x2, y2}.
using Box = std::array<float, 4>;

enum class TrackState { kNew, kTracked, kLost, kRemoved };

// A snapshot of a track returned to the caller.
struct Track {
  int id{-1};
  Box box{};                 // current xyxy estimate (from the Kalman mean)
  float score{0.0F};
  int label{-1};
  TrackState state{TrackState::kNew};
  int age{0};                // frames since first seen
  int hits{0};               // total matched detections
  int time_since_update{0};  // frames since last matched detection (coast count)
};

struct ByteTrackConfig {
  float track_high_thresh{0.6F};  // det >= this -> high-score (stage 1)
  float track_low_thresh{0.1F};   // [low, high) is the low band; below low dropped
  float new_track_thresh{0.7F};   // an unmatched high det must exceed this to spawn
  float match_thresh{0.8F};       // stage-1 gate: match kept iff IoU >= 1 - this
  float match_thresh_low{0.5F};   // stage-2 (low-score) gate
  int track_buffer{30};           // max frames a lost track may coast
  bool per_class{false};          // if true, only associate dets to same-label tracks
};

// Internal stateful track owner: the 8-dim constant-velocity Kalman filter plus
// the lifecycle bookkeeping. Declared here so tests can exercise the filter.
class KalmanBoxTracker {
 public:
  KalmanBoxTracker(const Box& xyxy, float score, int label, int id);

  void Predict();                             // advance one step (coast)
  void Update(const Box& xyxy, float score);  // correct with a measurement

  Box AsXyxy() const;  // mean[0..3] (cx,cy,a,h) -> xyxy
  Track View() const;

  int id{-1};
  int label{-1};
  float score{0.0F};
  TrackState state{TrackState::kNew};
  int age{0};
  int hits{0};
  int time_since_update{0};

 private:
  std::array<float, 8> mean_{};   // [cx, cy, a=w/h, h, vcx, vcy, va, vh]
  std::array<float, 64> cov_{};   // 8x8 row-major
};

// The tracker: feed one frame of detections per Update, get back the active tracks.
class ByteTracker {
 public:
  explicit ByteTracker(ByteTrackConfig cfg = {});

  // boxes: xyxy abs px; scores/labels are parallel arrays. Returns the currently
  // active tracks (confirmed + recently updated/coasting) with stable ids.
  std::vector<Track> Update(const std::vector<Box>& boxes, const std::vector<float>& scores,
                            const std::vector<int>& labels);
  void Reset();

 private:
  ByteTrackConfig cfg_;
  std::vector<KalmanBoxTracker> tracks_;
  int next_id_{1};
  int frame_{0};
};

}  // namespace detr::track
