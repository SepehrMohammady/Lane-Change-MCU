# Results explorer

One page for every dataset in the project: search fronts, on-board measurements,
quantization, five-seed retraining, the published comparisons and the scenario replays.
It is a single self-contained HTML file, so it opens offline and travels as one
attachment.

## Build

```
python dashboard/build.py            # dashboard/dist/results-explorer.html
python dashboard/build.py --share    # dashboard/dist/results-explorer-share.html
```

Standard library only. Rebuild after any new result; the page reads the result
files directly:

| view | source |
|---|---|
| Search | `datasets/*/results/nas-fronts/*.csv` |
| Hardware | `datasets/*/results/deploy/measurements.json`, resolved against `benchmarks_api.jsonl` |
| Quantization, Benchmark | the same registries and the published figures they cite |
| Seeds | `datasets/*/results/seeds/*.jsonl` (search recipe, same-recipe controls, highD re-rank) and the seeded hand-built runs in `datasets/*/logs/experiments.jsonl` |
| Scenario | the replay GIFs under `Materials/T4.5/` (local only) |
| exiD: Transfer, Seeds | `datasets/exid/results/transfer.jsonl`, `transfer_searched.jsonl`, `seeds/*.jsonl`, `prepared_meta.json` |

## Which file to send

The **share** build is the one to send outside the lab. It leaves out the project
status page, which lists internal to-dos, and the highD replay, which is drawn from
raw highD trajectories that the dataset licence does not let us pass on.

## Using it

- Dataset switcher at the top; views as tabs underneath.
- Hover any mark for its numbers; click a searched model for its measurements,
  sources and architecture.
- Every chart has a **Table** button with the same values.
- Links are shareable: `#highd/robust` (Seeds tab), or `#dmir/hardware/cls_best` to open an LCIR model.
- `?theme=light` or `?theme=dark` in the address forces a theme (useful on a projector).

## Adding a dataset

Add a `build_<name>()` in `build.py` returning the same keys as `build_highd()`, and
a `measurements.json` registry next to that dataset's deploy results. The exiD entry
is already there as a placeholder.
