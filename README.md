# 🎾 Tennis Analysis System

A computer-vision pipeline that tracks players and the ball in tennis footage, maps
pixels to real-world court coordinates, computes speeds and distances in physical
units, and — the differentiating feature — **classifies each shot as a serve,
forehand, backhand or volley** from the striking player's pose.

```
video ──▶ player tracking ──┐
     └──▶ ball tracking ────┼──▶ court homography ──▶ speeds & distances ──┐
     └──▶ court keypoints ──┘                                              │
                                                                           ▼
                          pose estimation ──▶ shot detection ──▶ stroke classification
                                                                           │
                                    annotated video + shot CSV/JSON + dashboard ◀┘
```

---

## Contents

- [Quick start](#quick-start)
- [What runs without training anything](#what-runs-without-training-anything)
- [Pipeline stages](#pipeline-stages)
- [Getting the datasets](#getting-the-datasets)
- [Training the models](#training-the-models)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [Dashboard](#dashboard)
- [Outputs](#outputs)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Design notes](#design-notes)

---

## Quick start

Requires **Python 3.11** (PyTorch and Ultralytics do not yet publish wheels for 3.13+).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run on a clip. Missing models degrade gracefully — see the next section.
python main.py --input input_videos/match.mp4 --output output/
```

The first run downloads the pretrained `yolo11n.pt` and `yolo11n-pose.pt` checkpoints
automatically (~10 MB).

---

## What runs without training anything

Three of the six model stages need weights you must train yourself. Rather than
refusing to start, the pipeline **disables the affected stage, warns clearly, and
completes with everything else intact**:

| Missing weights | What still works | What you lose |
|---|---|---|
| `models/ball_yolo11n.pt` | Player tracking, court mapping, player speeds/distances | Ball track, ball speed, shot detection, stroke labels |
| `models/court_keypoints_resnet18.pt` | Player + ball tracking in pixel space, stroke classification | Real-world units (km/h, metres), mini-court overlay |
| `models/stroke_classifier.joblib` | Everything through stage 5 | Stroke labels and the shot log |

So a fresh clone on a new clip produces an annotated video with tracked players
immediately, and each stage you train lights up more of the output.

---

## Pipeline stages

Each stage is a standalone module and can be exercised on its own.

**1. Video ingestion** — [`tennis_analysis/video_io.py`](tennis_analysis/video_io.py)
Frame extraction with FPS metadata preserved. Supports striding (`--stride`) and frame
caps (`--max-frames`); timestamps are always computed against the *source* FPS so
speeds stay correct regardless of sampling.

**2. Player detection & tracking** — [`tennis_analysis/detection.py`](tennis_analysis/detection.py)
Pretrained YOLO11 restricted to the COCO `person` class, with Ultralytics' built-in
ByteTrack for stable IDs. Ball kids, line judges and the umpire are filtered out by
projecting every track through the homography and keeping the two that spend the most
time inside the court boundary.

**3. Ball detection & tracking** — [`detection.py`](tennis_analysis/detection.py) + [`tracking.py`](tennis_analysis/tracking.py)
A YOLO11 model fine-tuned on tennis-ball data, smoothed by a constant-acceleration
Kalman filter that interpolates through occlusions and gates out false positives
(line markings, background clutter) that land implausibly far from the prediction.

**4. Court keypoints & homography** — [`tennis_analysis/court_keypoints/`](tennis_analysis/court_keypoints/)
A ResNet18 regressor predicts the 14 standard court line-intersections plus a
visibility score each. Confident keypoints are fitted via RANSAC to a pixel→metre
homography against the ITF reference layout, which drives both the real-world units
and the top-down mini-court overlay.

**5. Speed & distance** — [`tennis_analysis/analytics.py`](tennis_analysis/analytics.py)
Per-frame player and ball speed in km/h and cumulative distance in metres, computed in
court space over a smoothing window, with implausible values rejected as tracking noise.

**6. Stroke classification** — [`tennis_analysis/stroke_classification/`](tennis_analysis/stroke_classification/)
YOLO11-pose runs on each tracked player's crop. Shot moments are found from sharp
direction/speed changes in the ball trajectory; a ~0.5 s pose window around each is
converted into joint-angle and velocity features and classified. Two interchangeable
backends: a **RandomForest baseline** (default, CPU, trains in seconds) and a
**1D-CNN** over the same per-frame feature sequence.

**7. Output & analytics** — [`visualization.py`](tennis_analysis/visualization.py) + [`dashboard/app.py`](dashboard/app.py)
Annotated video, per-shot CSV/JSON, per-frame track log, and a Streamlit dashboard.

---

## Getting the datasets

None are bundled — they are large and carry their own licences.

```bash
# Roboflow datasets need a free API key: https://app.roboflow.com -> Settings -> API key
export ROBOFLOW_API_KEY=your_key_here

python scripts/download_datasets.py --dataset ball    # tennis-ball detection, YOLO format
python scripts/download_datasets.py --dataset court   # court keypoints, COCO format
python scripts/download_datasets.py --dataset thetis  # prints manual instructions
```

| Dataset | Source | Used for | Access |
|---|---|---|---|
| Tennis ball detection | [Roboflow Universe](https://universe.roboflow.com) | Stage 3 | API key |
| Tennis court keypoints | [Roboflow Universe](https://universe.roboflow.com) | Stage 4 | API key |
| THETIS | [thetis.image.ece.ntua.gr](http://thetis.image.ece.ntua.gr/) | Stage 6 | Request form |

If a Roboflow project has moved, browse Universe for a replacement and pass
`--workspace/--project/--version` explicitly.

**THETIS** is gated behind a request form, so `--dataset thetis` prints instructions
rather than downloading. Extract it to one directory per source class:

```
data/thetis/
  forehand_flat/   *.avi
  backhand_slice/  *.avi
  service_flat/    *.avi
  forehand_volley/ *.avi
  ...
```

THETIS ships finer-grained classes than the four this project predicts; they are folded
by a regex mapping. Print it with:

```bash
python scripts/train_stroke_classifier.py --show-label-map
```

---

## Training the models

### On Google Colab (recommended)

[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) trains all three models on
a free T4. Upload it to [Colab](https://colab.research.google.com), set
`Runtime → Change runtime type → T4 GPU`, and run Section 0 followed by whichever
section you need — they are independent.

The notebook mounts Google Drive and writes each model there as soon as it exists, so
an idle-timeout disconnect (~90 min on the free tier) doesn't cost you the run. It also
deliberately does **not** reinstall torch: Colab ships a CUDA-matched build, and
replacing it is the usual way to end up with a torch that cannot see the GPU.

Rough T4 timings: ball ~20–40 min, court ~15–30 min, strokes ~30–90 min (pose
extraction dominates, and is cached to Drive so retraining is instant).

### Locally

Both trainers accept the same `--device` values (`0`, `cuda`, `mps`, `cpu`); omit it to
auto-select. Apple Silicon works via `mps` but is several times slower than a T4.

### Ball detector (stage 3)

```bash
python scripts/train_ball_detector.py --data data/ball/data.yaml --epochs 60
# small ball, high-res footage? bump the input size:
python scripts/train_ball_detector.py --data data/ball/data.yaml --imgsz 1280 --model yolo11s.pt
```

Copies the best weights to `models/ball_yolo11n.pt`.

### Court keypoints (stage 4)

```bash
python scripts/train_court_keypoints.py --data data/court --epochs 50
```

Reports mean keypoint error in pixels each epoch — the number to watch. Writes the best
checkpoint to `models/court_keypoints_resnet18.pt`.

If your dataset orders keypoints differently from
[`KEYPOINT_NAMES`](tennis_analysis/court_keypoints/geometry.py), remap with
`--keypoint-order 0,1,3,2,...`.

### Stroke classifier (stage 6)

```bash
# RandomForest baseline — CPU, seconds
python scripts/train_stroke_classifier.py --data data/thetis --cache data/thetis_poses.npz

# 1D-CNN, same features and interface
python scripts/train_stroke_classifier.py --data data/thetis --backend cnn1d --cache data/thetis_poses.npz
```

Pose extraction dominates the runtime, so `--cache` is worth using: it makes retraining
and backend comparisons instant. Both backends print a validation confusion matrix.

---

## Configuration

Every threshold, model path and video setting lives in
[`config.yaml`](config.yaml) — no hardcoded constants. The schema is typed dataclasses
in [`tennis_analysis/config.py`](tennis_analysis/config.py), and **unknown keys raise**,
so a typo surfaces immediately instead of silently falling back to a default.

Things you are most likely to touch:

```yaml
ball:
  confidence: 0.15          # lower it if the ball is being missed
  kalman:
    max_age: 12             # frames to coast through an occlusion
    max_gate_distance: 150  # px; lower it to reject more false positives

court:
  refit_interval: 30        # frames between homography refits (fixed camera → raise it)
  max_reprojection_error_m: 1.0

stroke:
  min_direction_change_deg: 45.0   # lower → more shot candidates
  min_shot_interval_s: 0.4         # raise if one impact registers as several shots
  window_seconds: 0.25             # half-width of the pose window
```

---

## CLI reference

```
python main.py --input VIDEO [options]

  -i, --input PATH      input video file (required)
  -o, --output DIR      output directory (default: output)
  -c, --config PATH     config YAML/JSON (default: config.yaml)
      --stroke          force stroke classification on
      --no-stroke       skip stroke classification (stages 1-5 only)
      --dashboard       launch the Streamlit dashboard when the run finishes
      --max-frames N    process at most N frames (smoke tests)
      --stride N        process every Nth frame
  -v, --verbose         debug logging
```

Examples:

```bash
# Quick smoke test on the first 200 frames
python main.py -i clip.mp4 -o output/ --max-frames 200 --no-stroke

# Full run, then open the dashboard
python main.py -i match.mp4 -o output/ --dashboard

# Halve the compute on long footage
python main.py -i match.mp4 -o output/ --stride 2
```

---

## Dashboard

```bash
streamlit run dashboard/app.py -- --results output/
```

Shows shot-type distribution per player, speed over time with shot moments marked,
ball speed per shot, and total distance covered per player.

---

## Outputs

Written to `--output`:

| File | Contents |
|---|---|
| `annotated.mp4` | Player/ball boxes + IDs, speed readouts, stroke labels, mini-court overlay |
| `shots.csv` / `shots.json` | Per shot: frame, timestamp, player, stroke, confidence, ball speed, player speed |
| `tracks.csv` | Per frame per entity: court coordinates (m), speed (km/h), cumulative distance (m) |
| `player_summary.csv` | Per player: total distance, average and max speed |

---

## Project structure

```
tennis_analysis/
├── config.py                    # typed config schema (rejects unknown keys)
├── types.py                     # BBox, PlayerDetection, BallDetection, Shot
├── video_io.py                  # stage 1: frame extraction, FPS metadata, writing
├── detection.py                 # stages 2-3: YOLO11 player + ball detectors
├── tracking.py                  # stage 3: ball Kalman filter and interpolation
├── analytics.py                 # stage 5: speed, distance, CSV/JSON export
├── visualization.py             # stage 7: overlay drawing
├── pipeline.py                  # orchestration
├── court_keypoints/             # stage 4
│   ├── geometry.py              #   ITF layout, homography, court-boundary test
│   ├── model.py                 #   ResNet18 regressor + loss
│   ├── detector.py              #   inference + homography maintenance
│   └── mini_court.py            #   top-down overlay
└── stroke_classification/       # stage 6
    ├── pose.py                  #   YOLO11-pose over player crops
    ├── shot_detection.py        #   shot moments from ball trajectory
    ├── features.py              #   pose → joint angles, velocities, aggregates
    └── classifier.py            #   RandomForest + 1D-CNN behind one interface

scripts/     download_datasets.py, train_ball_detector.py,
             train_court_keypoints.py, train_stroke_classifier.py
notebooks/   colab_train.ipynb   trains all three models on a free Colab T4
dashboard/   app.py              Streamlit dashboard
tests/       geometry, kinematics, stroke features, tracking, classifier
main.py                          CLI entry point
config.yaml                      all thresholds and model paths
```

---

## Tests

```bash
pytest tests/ -q          # 165 tests, ~5 s
```

Coverage concentrates on the logic that is verifiable in isolation, using analytically
known ground truth rather than eyeballed expectations:

- **`test_geometry.py`** — a synthetic camera projects the 14 reference points through
  a known homography; fitting on those projections must recover court metres exactly.
  Also covers confidence gating, degenerate/collinear inputs and court-boundary logic.
- **`test_kinematics.py`** — a player moving at a known m/s must yield exactly the
  expected km/h and metres; covers gaps, ID-switch teleports and speed clipping.
- **`test_stroke_features.py`** — a straight arm must measure 180°, a raised arm must
  read above the shoulder, and a player scaled or moved across court must featurise
  identically. Plus shot-moment detection and window bounds.
- **`test_tracking.py`** — Kalman smoothing beats raw noise, gaps interpolate, outliers
  are gated, dead tracks reseed cleanly.
- **`test_stroke_classifier.py`** — both backends learn synthetic separable strokes and
  survive a save/load round trip.

---

## Design notes

**Why two passes over the video.** Several outputs depend on the whole sequence:
picking the two on-court players needs every track's court-time, and a shot moment is
only identifiable from the ball trajectory *after* impact. A single streaming pass
would have to draw labels it does not yet know. Pass 1 runs inference and collects
observations; pass 2 re-reads the video and renders. Re-reading is cheap next to
inference.

**Why the feet, not the box centre.** Player positions project from the bottom-centre
of the bounding box. The box centre floats at chest height and lands metres deep into
the court once passed through a ground-plane homography.

**Why a visibility head on the keypoint model.** Broadcast framing routinely crops the
far baseline out of shot. Without a visibility output, the model is forced to
hallucinate positions for keypoints it cannot see, and those hallucinations poison the
homography. Low-confidence keypoints are dropped before the RANSAC fit.

**Why the homography is refit periodically, not per frame.** For fixed-camera footage
the court is near-static. Refitting every 30 frames and reusing the last *accepted* fit
in between saves compute and, more importantly, means a brief occlusion of the baseline
does not knock out speed measurement for a whole rally.

**Why pose runs on crops.** On broadcast footage a player occupies a small fraction of
the frame. Cropping first raises effective resolution on the limbs that carry the
stroke signal, and sidesteps matching whole-frame pose detections back to track IDs.

**Why the shot-detection score squashes the speed term.** Direction reversal and speed
change both indicate a strike, but an unbounded speed term outvotes a genuine 180°
reversal and peaks a frame or two *before* impact — where the ball is decelerating but
has not yet turned around. The pose window centres on this index, so that offset
matters. The speed contribution is `tanh`-squashed to at most half the direction term.

---

## Known limitations

- **Fixed-camera assumption.** The periodic homography refit suits a static broadcast
  camera. Heavy panning or cuts need `court.refit_interval: 1`.
- **Two players only.** Player selection keeps the top two tracks; doubles is not
  supported.
- **Stroke labels inherit THETIS's domain.** THETIS is staged, single-player footage.
  Accuracy on live broadcast angles will be lower than the validation number suggests —
  treat the reported figure as an upper bound.
- **Handedness is not modelled.** Features are computed for both arms, so the
  classifier must infer the racket hand from the data rather than being told.

---

## License

[MIT](LICENSE).

Note that the datasets and pretrained checkpoints this project *uses* carry their own
licences — Ultralytics YOLO11 weights are AGPL-3.0, and the Roboflow and THETIS
datasets have their own terms. Check them before any commercial use.
# tennis_analysis
