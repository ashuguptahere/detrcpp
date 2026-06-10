// Copyright 2026 detrcpp authors. Apache-2.0.
//
// ByteTrack: Kalman filtering + two-stage IoU association give stable ids on a
// moving box, persist ids through short occlusion gaps, recover low-score
// detections in stage 2, and reap tracks after the coast buffer expires.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "detr/track/bytetrack.hpp"

namespace detr::track {
namespace {

// A 40x40 box translating +10px/frame in x: frame i -> [10i, 0, 40+10i, 40].
Box MovingBox(int i) {
  const float x = 10.0F * static_cast<float>(i);
  return {x, 0.0F, x + 40.0F, 40.0F};
}

TEST(ByteTrack, StableIdOnMovingBox) {
  ByteTracker tr;
  int id = -1;
  for (int f = 0; f < 8; ++f) {
    auto out = tr.Update({MovingBox(f)}, {0.9F}, {0});
    ASSERT_EQ(out.size(), 1U) << "frame " << f;
    if (f == 0) {
      id = out[0].id;
    }
    EXPECT_EQ(out[0].id, id) << "frame " << f;        // same id throughout
    EXPECT_NEAR(out[0].box[0], MovingBox(f)[0], 8.0F);  // tracks input within Kalman lag
  }
}

TEST(ByteTrack, IdPersistsThroughOcclusionGap) {
  ByteTracker tr;  // default track_buffer = 30
  int id = -1;
  for (int f = 0; f < 3; ++f) {
    auto out = tr.Update({MovingBox(f)}, {0.9F}, {0});
    ASSERT_EQ(out.size(), 1U);
    id = out[0].id;
  }
  for (int f = 3; f < 6; ++f) {
    auto out = tr.Update({}, {}, {});  // occluded: no detections (coast)
    EXPECT_TRUE(out.empty()) << "frame " << f;
  }
  auto out = tr.Update({MovingBox(6)}, {0.9F}, {0});  // reappears near the predicted spot
  ASSERT_EQ(out.size(), 1U);
  EXPECT_EQ(out[0].id, id);  // re-associated to the SAME id (no new track)
}

TEST(ByteTrack, LowScoreDetectionRecoveredInStage2) {
  ByteTracker tr;
  int id = -1;
  for (int f = 0; f < 3; ++f) {
    id = tr.Update({MovingBox(f)}, {0.9F}, {0})[0].id;
  }
  // A low-score detection (in [low, high)) must still be matched via stage 2.
  auto out = tr.Update({MovingBox(3)}, {0.3F}, {0});
  ASSERT_EQ(out.size(), 1U);
  EXPECT_EQ(out[0].id, id);
}

TEST(ByteTrack, NewTrackOnlyAboveSpawnThreshold) {
  ByteTracker tr;  // new_track_thresh = 0.7
  // A high-band but below-spawn detection (0.65) must NOT create a track.
  auto out = tr.Update({MovingBox(0)}, {0.65F}, {0});
  EXPECT_TRUE(out.empty());
  // Above the spawn threshold -> a track appears.
  out = tr.Update({MovingBox(1)}, {0.9F}, {0});
  EXPECT_EQ(out.size(), 1U);
}

TEST(ByteTrack, CoastExpiresAfterBuffer) {
  ByteTrackConfig cfg;
  cfg.track_buffer = 2;
  ByteTracker tr(cfg);
  int id = -1;
  for (int f = 0; f < 2; ++f) {
    id = tr.Update({MovingBox(f)}, {0.9F}, {0})[0].id;
  }
  for (int f = 2; f < 6; ++f) {  // coast longer than the buffer
    tr.Update({}, {}, {});
  }
  auto out = tr.Update({MovingBox(6)}, {0.9F}, {0});
  ASSERT_EQ(out.size(), 1U);
  EXPECT_NE(out[0].id, id);  // the old track expired; this is a fresh id
}

TEST(ByteTrack, KalmanPredictUpdateConverges) {
  const Box b{10.0F, 10.0F, 50.0F, 50.0F};
  KalmanBoxTracker k(b, 0.9F, 0, 1);
  for (int i = 0; i < 6; ++i) {
    k.Predict();
    k.Update(b, 0.9F);  // repeated identical measurements
  }
  const Box est = k.AsXyxy();
  for (std::size_t j = 0; j < 4; ++j) {
    EXPECT_NEAR(est[j], b[j], 1.0F);  // mean stays at the stationary box
  }
}

}  // namespace
}  // namespace detr::track
