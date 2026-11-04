# Results explorer

One page for every dataset in the project: search fronts, on-board latency and energy,
quantization, five-seed retraining, the published comparisons and the scenario replays.
It opens on a Study page that states the problem, the idea and the method before any
result (supervisor's suggestion, 2026-10-30).
It is a single self-contained HTML file, so it opens offline and travels as one
attachment.

## Build

```
python dashboard/build.py            # dashboard/dist/results-explorer.html
python dashboard/build.py --lab      # dashboard/dist/results-explorer-lab.html
python dashboard/build.py --share    # dashboard/dist/results-explorer-share.html
```

Standard library only. Rebuild after any new result; the page reads the result
files directly:

| view | source |
|---|---|
| Search | `datasets/*/results/nas-fronts/*.csv`; for the final searches, the 12 kept candidates in `seeds/select_*.jsonl` (mean of five runs) |
| Hardware | `datasets/*/results/deploy/measurements.json`, resolved against `benchmarks_api.jsonl`; own-board latency from `local_bench.jsonl`, energy per inference from `energy_local.jsonl` |
| Quantization, Benchmark | the same registries and the published figures they cite |
| Seeds | `datasets/*/results/seeds/*.jsonl` (search recipe, same-recipe controls, the model each later search kept, highD re-rank) and the seeded hand-built runs in `datasets/*/logs/experiments.jsonl` |
| Scenario | the replay GIFs under `Materials/T4.5/` (local only) |
| exiD: Transfer, Seeds | `datasets/exid/results/transfer.jsonl`, `transfer_searched.jsonl`, `seeds/*.jsonl`, `prepared_meta.json` |

## Which file to send

| build | status page | highD and exiD replays | for |
|---|---|---|---|
| `results-explorer.html` | yes | yes | us |
| `results-explorer-lab.html` | no | yes | colleagues in the lab |
| `results-explorer-share.html` | no | no | anyone outside the lab |

The status page lists internal to-dos. The highD and exiD replays are drawn from raw
trajectories, which the dataset licences do not let us pass on outside the lab.

## Using it

- Study first: the four requirements of an in-vehicle predictor, where it is used, the idea,
  the pipeline and the changes to µNAS, then four results that link into the datasets.
- Dataset switcher at the top; views as tabs underneath. Colour follows the dataset
  (blue LCIR, orange highD, green exiD); the pages that cover all of them (Study,
  Compare) are plum.
- Hover any mark for its numbers; click a searched model for its measurements,
  sources and architecture.
- Every chart has a **Table** button with the same values.
- Links are shareable: `#highd/robust` (Seeds tab), or `#dmir/hardware/cls_best` to open an LCIR model.
- `?theme=light` or `?theme=dark` in the address forces a theme (useful on a projector).

## Adding a dataset

Add a `build_<name>()` in `build.py` returning the same keys as `build_highd()`, and
a `measurements.json` registry next to that dataset's deploy results.
