# 🎾 Tennis Analysis System

Computer-vision pipeline for tennis footage: tracks players and the ball, maps pixels to
real-world court coordinates via homography, computes speeds and distances in physical
units, and classifies each shot as a **serve, forehand, backhand or volley** from the
striking player's pose.

Built with YOLO11 + ByteTrack, a ResNet court-keypoint regressor, a Kalman-smoothed ball
track, and pose-based stroke classification.

---

## Quick start

Requires **Python 3.11** (PyTorch and Ultralytics publish no wheels for 3.13+).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py --input input_videos/match.mp4 --output output/
```

Pretrained `yolo11n.pt` and `yolo11n-pose.pt` download automatically on first run.

---

## Models

Three stages need trained weights. Missing ones are **disabled with a clear warning**
rather than crashing the run, so a fresh clone produces useful output immediately:

| Missing | Still works | Lost |
|---|---|---|
| ball detector | player tracking, court mapping, player speeds | ball track, shot detection |
| court keypoints | tracking in pixel space | real-world units, mini-court overlay |
| stroke classifier | everything **including shot-moment detection** | stroke *names* — shots log as `unclassified` |

Shot moments come from the ball trajectory alone, so even without a classifier you get a
timestamped shot log with speeds — enough to bootstrap a labelled set from your own footage.

### Training

**[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)** trains all three on a free
Colab T4 — roughly 20–40 min (ball), 15–30 min (court), 30–90 min (strokes). Sections are
independent; weights save to Drive as they finish.

Locally, via `scripts/train_{ball_detector,court_keypoints,stroke_classifier}.py` — all
accept `--device 0|cuda|mps|cpu`.

Datasets are not bundled:

```bash
python scripts/download_datasets.py --dataset ball|court|thetis
```

Roboflow needs a free API key; THETIS is gated behind a [request form](http://thetis.image.ece.ntua.gr/)
and prints instructions instead of downloading.

### Using weights you already have

Point [`config.yaml`](config.yaml) at them. Ball models load through Ultralytics (YOLOv5u
/ v8 / v11). Court models are fingerprinted automatically — `architecture: auto` handles
both this project's dual-head ResNet18 and single-head `ResNet{18,34,50} → fc(28)`
regressors.

> **⚠️ Verify keypoint ordering first.** The homography maps keypoint *i* to a fixed court
> location. If your checkpoint emits them in a different order **nothing throws** — RANSAC
> fits a plausible-looking homography and every distance is silently wrong. Corner-based
> sanity checks won't catch it either. Reprojection error is the diagnostic:
>
> ```python
> kps, conf = CourtKeypointDetector(Config.load("config.yaml").court).detect(frame)
> print(CourtHomography.from_keypoints(kps, conf).reprojection_error)   # want < 0.1 m
> ```
>
> Several metres means the order differs — set `court.keypoint_order` to the permutation
> mapping canonical index → checkpoint index. One widely circulated checkpoint needs
> `[0,1,2,3,4,6,5,7,8,9,10,11,12,13]`; that single swap moved error from **3.61 m to 0.02 m**.

---

## Usage

```
python main.py -i VIDEO [-o DIR] [-c CONFIG]
               [--no-stroke] [--dashboard] [--max-frames N] [--stride N] [-v]
```

Outputs written to `-o`:

| File | Contents |
|---|---|
| `annotated.mp4` | player/ball boxes + IDs, speed readouts, stroke labels, mini-court |
| `shots.csv` / `.json` | per shot: frame, timestamp, player, stroke, confidence, ball + player speed |
| `tracks.csv` | per frame: court coordinates (m), speed (km/h), cumulative distance |
| `player_summary.csv` | per player: total distance, average and max speed |

Dashboard (shot distribution, speed over time, distance per player):

```bash
streamlit run dashboard/app.py -- --results output/
```

---

## Configuration

Every threshold and path lives in [`config.yaml`](config.yaml) — no hardcoded constants.
The schema is typed dataclasses in [`config.py`](tennis_analysis/config.py), and unknown
keys raise, so a typo surfaces immediately instead of silently defaulting.

Most commonly tuned:

```yaml
ball:
  confidence: 0.15            # lower if the ball is being missed
  kalman: {max_age: 12}       # frames to coast through an occlusion
court:
  refit_interval: 30          # fixed camera → raise; panning/cuts → set to 1
stroke:
  min_direction_change_deg: 45.0   # lower → more shot candidates
  min_shot_interval_s: 0.4         # raise if one impact registers as several
```

---

## Structure

```
tennis_analysis/
├── video_io.py             frame extraction, FPS metadata, writing
├── detection.py            YOLO11 player (+ByteTrack) and ball detectors
├── tracking.py             ball Kalman filter, gap interpolation, outlier gating
├── analytics.py            speed, distance, CSV/JSON export
├── visualization.py        overlay drawing
├── pipeline.py             orchestration
├── court_keypoints/        ITF layout, homography, ResNet regressor, mini-court
└── stroke_classification/  pose, shot detection, features, RF + 1D-CNN backends

scripts/     dataset download + three training scripts
notebooks/   colab_train.ipynb
dashboard/   Streamlit app
tests/       193 tests
```

---

## Tests

```bash
pytest tests/ -q          # 193 tests, ~11 s
```

Ground truth is analytic rather than eyeballed: a synthetic camera projects the 14
reference points through a known homography and fitting must recover court metres exactly;
a player moving at a known m/s must yield exactly the expected km/h; a straight arm must
measure 180°.

---

## Notes & limitations

- **Two passes over the video.** Selecting the two on-court players needs every track's
  court-time, and a shot moment is only identifiable from the trajectory *after* impact —
  a single streaming pass would have to draw labels it does not yet know.
- **Positions project from the feet**, not the box centre; the centre floats at chest
  height and lands metres deep through a ground-plane homography.
- **Fixed camera assumed.** Heavy panning or cuts need `court.refit_interval: 1`.
- **Singles only** — player selection keeps the top two tracks.
- **Stroke accuracy on broadcast angles will trail THETIS validation numbers**, which come
  from staged single-player footage. Treat the reported figure as an upper bound.

---

## License

[MIT](LICENSE). The models and datasets this project *uses* carry their own terms —
Ultralytics YOLO11 weights are AGPL-3.0. Check before commercial use.
