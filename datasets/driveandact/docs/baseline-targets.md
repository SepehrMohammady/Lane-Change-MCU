# Drive&Act — what we would actually be competing against

Source: Martin et al., *Drive&Act*, ICCV 2019 (Table 2) and Peng et al.,
*TransDARC*, IROS 2022 (Table I). Both PDFs in `Materials/Drive&Act/`.
Metric: mean per-class top-1 accuracy, **averaged over the 3 official splits**.
Random baseline = 2.94% (1/34).

## Fine-grained activities (34 classes) — the benchmark task

| Track | Model | Val | **Test** |
|---|---|--:|--:|
| **Pose** | Interior | 45.23 | 40.30 |
| **Pose** | Pose | 53.17¹ | 44.36 |
| **Pose** | Two-Stream | 53.76 | 45.39 |
| **Pose** | **Three-Stream** | 55.67 | **46.95** ← best pose |
| Video | C3D | 49.54 | 43.41 |
| Video | P3D ResNet | 55.04 | 45.32 |
| Video | I3D | 69.57 | 63.64 |
| Video | CTA-NET | 72.42 | 65.25 |
| Video | Video Swin | 88.10 | 85.74 |
| Video | **TransDARC** | 93.58 | **89.65** ← current SOTA |

¹ ICCV reports 53.17 val; TransDARC's reproduction of the same row reports
55.17. Test agrees at 44.36 in both. Cite the original and use test.

## The strategic picture, stated plainly

**The pose track has not moved since 2019.** Every advance since then —
I3D → CTA-NET → Video Swin → TransDARC — is on **video**, and video has gone
from 63.6% to 89.7%. Nobody has revisited the skeleton track with modern
methods, and nobody has done NAS or MCU deployment on it at all.

That is the opening, and it also bounds what we may promise:

- **Not achievable:** beating 89.65%. A skeleton model on a Cortex-M7 cannot
  compete with a pretrained video transformer, and claiming otherwise would be
  the same modality-mismatch error we removed from the DMIR paper.
- **The honest target:** the pose track, i.e. **46.95%** (Three-Stream) or
  **45.39%** (Two-Stream, pose-only without the interior stream).
- **The contribution:** competitive pose-based accuracy at microcontroller
  cost, which is unexplored territory here — not a new dataset SOTA.

Worth noting for the framing: the pose Three-Stream (46.95%) already **beats
two of the three video CNNs** — C3D (43.41%) and P3D (45.32%). Skeletons are
not obviously the weak modality; they are the under-investigated one.

## Realism about the difficulty

Beating 46.95% is **not** a formality. Only 6,642 training chunks across 34
classes (~195 per class), so overfitting is the dominant risk and the search's
validation signal will be noisy. The 2019 baselines are LSTM two/three-stream
models already tuned for exactly this regime. Treat "match the pose baselines
at a fraction of the cost" as the realistic outcome and anything above as upside.

## Practical notes for the adapter

- The paper builds its pose baselines on **13 upper-body joints** (OpenPose,
  triangulated from 3 frontal views). The released `openpose_3d` CSVs carry
  **26** joint groups (BODY_25 + background), including lower-body joints that
  are occluded for a seated driver and are frequently zero. Prefer the 13
  upper-body joints for comparability; test the full set as a variant.
- Missing detections appear as exact `0.0` across a joint's x/y/z/p. The paper
  interpolates from neighbouring frames — match that, do not leave zeros.
- Each joint has a confidence `p`. Either drop it (13×3 = 39 channels) or keep
  it as an input (13×4 = 52). Worth an ablation.
- The Three-Stream baseline additionally consumes the **car-interior** features
  (`iccv_interior.zip`, already downloaded): distances from hands and head to
  interior objects. A pose-only comparison should target Two-Stream (45.39%).
- Protocol: report the **mean over the 3 splits**, so budget 3× evaluation.
