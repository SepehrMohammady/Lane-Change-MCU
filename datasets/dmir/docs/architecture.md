# Discovered architectures (what the NAS actually found)

Decoded from the ST Edge AI model graphs in `datasets/dmir/results/deploy/*.svg`. TFLite
expresses Conv1D as Conv2D with a dummy dimension, so the `ExpandDims`/`Reshape`
pairs are plumbing, not layers; below they are collapsed into 1D terms.

## Reference CNN (`cnn_multi`, 441 k params, 91.69%)

```
Input 50×31
  → Conv1D(64, k=3) + ReLU + BatchNorm      → 50×64
  → Conv1D(128, k=3) + ReLU + BatchNorm     → 50×128
  → Flatten                                 → 6400
  → Dense(64) + ReLU
  → Dense(3) → Softmax
```

Temporal resolution is held at the full 50 steps all the way to the flatten.

## Ours (`cls_best`, 84 k params, 92.08%)

```
Input 50×31
  → MaxPool1D                               → 25×31
  → DWConv1D (stride 2)                     → 13×31
  → Conv1D 1×1 (31→77) + ReLU               → 13×77
  → DWConv1D + ReLU
  → DWConv1D (stride 2) + ReLU              → 7×77
  → Conv1D 1×1 + ReLU
  → Add                ← residual connection
  → AvgPool1D                               → 4×77
  → Flatten                                 → 308
  → FC(223) + ReLU → FC(23) + ReLU → FC(3)
```

The search independently arrived at three MCU-friendly choices:
1. **depthwise-separable convolutions** (DWConv1D followed by 1×1 Conv1D) — the
   canonical efficient-CNN pattern, not hand-specified;
2. a **residual connection** (`Add`);
3. **aggressive early downsampling** — 50 → 25 → 13 → 7 → 4.

## Ours (`cls_tiny`, 8 k params, 91.30%, 0.79 ms)

Only four real layers:

```
Input 50×31
  → Conv1D 1×1 (31→14) + ReLU               → 50×14   ← channel squeeze first
  → DWConv1D (k=3, stride 2) + ReLU         → 25×14
  → AvgPool1D                               → 7×14
  → Flatten                                 → 98
  → FC(73) + ReLU → FC(3)
```

Same recipe as `cls_best`, pushed harder: squeeze channels immediately with a
1×1 conv (31→14), one cheap depthwise conv for temporal structure, aggressive
pooling (50→25→7), and a small head flattening just 98 features. Its flash is
again dominated by the first FC (`gemm_13`, 73×98 ≈ 28.6 KB of 37 KB).

## Ours (`lcr_best`, 117 k params, MAE 0.287, 14.06 ms) — a two-branch DAG

Unlike the classifiers, the regression model is **not a chain**: the input feeds
two parallel branches that merge with `Add`.

```
                ┌─ Conv1D(116, k=5, s2) + BN + ReLU → 25×116
                │    → Conv1D 1×1 (116→11) + BN + ReLU → 25×11
                │    → MaxPool1D → 7×11 ─────────────────────────┐
Input 50×31 ────┤                                                 ├─ Add → 7×11
                └─ MaxPool1D → 25×31                              │
                     → DWConv1D(k=5, s2) + ReLU → 13×31           │
                     → Conv1D(11, k=5) + BN → 7×11 ───────────────┘
  → Conv1D(95, k=5) + ReLU        → 7×95
  → Conv1D(76, k=5) + BN + ReLU   → 7×76
  → MaxPool1D                     → 3×76
  → Conv1D 1×1 (92) + BN + ReLU   → 3×92
  → MaxPool1D                     → 2×92
  → Flatten(184) → FC(249) + ReLU → FC(1)
```

Two observations:

- **Task shapes topology.** The search space permits parallel branches for every
  task, but only regression used them; the classifiers stayed essentially
  sequential. Multi-path structure was discovered where it helped, not imposed.
- **Flash is spread, not FC-dominated.** `gemm_37` (185 KB, 39.6% of weights),
  `conv2d_22` (145 KB) and `conv2d_1` (72 KB) together are ~85% of the 474 KB —
  a different profile from the classifiers, where one FC holds 82.5–90.9%.

## Ours (`lcl_best`, 106 k params, MAE 0.317, 28.77 ms) — a three-branch DAG

The most parallel model the search produced: the input fans out to **three**
paths, merged by a single 3-way `Add` (which TFLite lowers to two binary Adds).

```
                ┌─ A: Conv1D 1×1 (31→65) + BN + ReLU → 50×65
                │      → Conv1D(33, k=5, s2) + BN + ReLU → 25×33 ──┐
                │                                                   │
Input 50×31 ────┼─ B: Conv1D 1×1 (31→60) + BN  [no ReLU] → 50×60    ├─ Add → 25×33
                │      → Conv1D(33, k=5, s2) + BN + ReLU → 25×33 ──┤
                │                                                   │
                └─ C: MaxPool1D → 25×31                             │
                       → Conv1D(33, k=5) + BN + ReLU → 25×33 ──────┘
  → Conv1D(86, k=3, s2) + BN  [no ReLU]  → 13×86
  → Conv1D 1×1 (86→113) + BN + ReLU      → 13×113
  → Conv1D 1×1 (113→116) + ReLU          → 13×116
  → DWConv1D(k=3) + BN + ReLU            → 13×116
  → Conv1D(72, k=3) + BN  [no ReLU]      → 13×72
  → Conv1D 1×1 (72→60) + ReLU            → 13×60
  → MaxPool1D(4) → 4×60 → Flatten(240) → FC(49) + ReLU → FC(1)
```

Decode verified two ways: our computed weights total **403.6 KiB against ST's
measured 403.61 KiB (0.00%)**, and TFLite params 103,325 + 2,444 folded BatchNorm
params (4 × 611 channels) = **105,769 exactly**.

- **Flash is conv-dominated, uniquely.** `conv2d_30` (100,512 B, 24.3%) beats the
  first FC (47,236 B, 11.4%). Every other model we have is FC-dominated.
- **RAM is liveness-bound.** See deployment.md — the three-branch merge pins the
  input plus two small branch outputs across many ops, which the chain rule
  cannot express.

## What the search converges to (pattern across models)

All searched models independently adopt the same core moves:
1. **1×1 convolutions to re-shape the channel dimension** (squeeze 31→14 in the
   tiny model, 116→11 in the regression branch; expand 31→77 in cls_best);
2. **depthwise convolutions** for temporal structure (cheap per-channel filters);
3. **aggressive early downsampling** (50→25→7, 50→25→13→7→4, or 50→25→7→3→2);
4. **a small flattened head** — 98, 184, 240 or 308 features, versus the
   reference's 6400.

**Task shapes topology.** The search space permits parallel branches for every
task, yet only the regression models used them — and they used them in
proportion to difficulty:

| model | task | branches | merges | biggest-FC share of flash |
|---|---|--:|--:|--:|
| cls_tiny | 3-class | 1 (chain) | 0 | 90.9% |
| cls_best | 3-class | 1 (+1 residual) | 1 | 82.5% |
| lcr_best | TTLC regression | 2 | 1 | 39.6% |
| lcl_best | TTLC regression | **3** | 2 (one 3-way) | **11.4%** |

The classifiers stayed essentially sequential and FC-dominated; the regression
models fanned out and shifted their budget into convolutions. Nothing in the
search space required this.

## RAM = max(input floor, widest layer)

`cls_best` (84 k) and `cls_tiny` (8 k) both measure ~9.2 KB RAM despite a 9×
parameter gap: the 50×31 float32 input tensor is 6.2 KB by itself, a floor no
architecture can beat in FP32; activations add only ~2–3 KB.

`lcr_best` shows the other regime: 20.8 KB, because its 116-channel branch emits
25×116×4 B = 11.6 KB, so the peak working set ≈ 6.2 + 11.6 ≈ 17.8 KB (measured
18.91 KiB). So RAM is set by **layer width**, not depth or parameter count, and
is floored by the input tensor. int8 input would cut the floor to 1.55 KB. This
is why the RAM advantage (~4×) is far smaller than the flash advantage (5–47×).

## Why ours is 5.2× smaller and 9.2× faster (quantified)

Both models spend most of their flash in the fully-connected head. The
difference is what enters that head:

| | Reference | Ours | ratio |
|---|--:|--:|--:|
| feature map into the head | 50×128 = **6400** | 4×77 = **308** | 20.8× |
| first FC shape | 64×6400 = 409,600 | 223×308 = 68,684 | 6.0× |
| that layer's flash | ~1.64 MB (~95% of model) | ~280 KB (~82% of model) | 5.9× |

So the honest framing is **not** "we avoid a large FC" — ours is dense-dominated
too. It is that pooling *before* the head shrinks the FC input 20.8×, which buys
a **wider** head (223 units vs 64) at ~6× less flash. Downsampling also cuts the
conv work, giving the 12.4× MACC reduction and the 9.2× measured speed-up.

Runtime profile differs as well: the reference's time is concentrated in the
`eltwise` BatchNorm ops over its 50×128 tensors, while ours is spread across
convs/pools — its big FC costs almost no time. Flash-bound and time-bound layers
are decoupled in both models.
