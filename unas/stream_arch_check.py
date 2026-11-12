"""Does the architecture matter when the training windows come from whole drives? (2026-11-12)

Under the lab's prepared windows every shortlisted network of our searches reaches about the same
accuracy (90.4 to 91.6% on the test set), so a search could only lower cost. A search "under the
whole-drive measure" (training on windows taken from whole drives, unas/stream_train_check.py) makes
sense only if that measure tells architectures apart. This check trains a ladder of existing networks on
such windows and scores them on the whole drives of the validation drivers. The test drivers are not
touched. Fixed before any run (local notes, 2026-11-12 15:20):

  data        the lab's split. Training: windows from whole drives (every frame from the 50th on that is
              not ignored, class of its last frame, balanced by subsampling, rng 0) of the sessions with
              lane changes of the 35 different training drives (the 36 training files without user 24,
              a copy of user 23); early stopping on the windows of the seven validation drivers built
              the same way.
  candidates  seven networks from the shortlists of our searches, by cost (14.4k to 152.3k MACs), and
              two hand-designed references if their files load: the lab's reference CNN (441k
              parameters; its softmax output layer is made linear, the recipe's loss takes logits) and
              our hand-designed network of July (84k parameters).
  recipe      as unas/input_scaling_check.train (Adam, at most 80 epochs, early stopping on validation
              accuracy, patience 12), seeds 100-104; int8 as deployed (500 calibration windows, rng 42,
              limited to +-4), scored as int8 on the PC.
  measure     share of correct frames in the validation drivers' sessions with lane changes, mean over
              the seeds.
  rule        GO for a search on windows from whole drives if the best candidate is at least 2 points
              above the deployed network and the difference is at least twice its standard error (seed
              spread of the two). Otherwise NO GO: the measure does not tell the candidates apart.

Run in the WSL dmir_nas venv, from the fork:
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data SEEDS=100,101,102,103,104 \\
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/stream_arch_check.py [candidate index ...]
  .venv\\Scripts\\python unas/stream_arch_check.py --report

Output: datasets/dmir/results/seeds/stream_arch.jsonl (one line per candidate and seed; resumable) and
datasets/dmir/results/hil/stream_arch.json. The networks are kept in datasets/dmir/data/stream_arch/.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_cv_check as cv                                 # noqa: E402  (numpy only)

stc, dr = cv.stc, cv.dr
NV = REPO / "datasets" / "dmir" / "results" / "nas-v2"
DEPLOY = REPO / "datasets" / "dmir" / "results" / "deploy"
KEEP = REPO / "datasets" / "dmir" / "data" / "stream_arch"
OUT = REPO / "datasets" / "dmir" / "results" / "seeds" / os.environ.get("OUT_NAME", "stream_arch.jsonl")
SUMMARY = REPO / "datasets" / "dmir" / "results" / "hil" / f"{OUT.stem}.json"
TRAIN = [u for u in range(1, 51) if u not in (2, 7, 13, 18, 25, 31, 36, 5, 8, 10, 12, 16, 19, 27, 24)]
VAL = [5, 8, 10, 12, 16, 19, 27]
DEPLOYED = "30.4k MACs, 2,336 parameters (deployed)"
CANDIDATES = (                                               # (name, file, MACs of our cost model or None)
    ("14.4k MACs, 2,132 parameters", NV / "dmir_cls_v4_k3_single_s42" / "model_1793742903250608262_3303363.h5", 14360),
    (DEPLOYED, NV / "dmir_cls_v4_k3_single_s42" / "model_1793742305537376579_3303354.h5", 30448),
    ("49.1k MACs, 7,119 parameters", NV / "dmir_cls_v4_k3_single_s42" / "model_1793742338466964583_3303363.h5", 49091),
    ("68.0k MACs, 3,424 parameters", NV / "dmir_cls_v4_k3_single_s43" / "model_1793874640541565910_600320.h5", 68023),
    ("85.1k MACs, 4,288 parameters", NV / "dmir_cls_v4_k3_single_s43" / "model_1793873974197815374_488437.h5", 85086),
    ("146.0k MACs, 6,528 parameters", NV / "dmir_cls_v4" / "model_1790355881847038030_1029276.h5", 145986),
    ("152.3k MACs, 9,660 parameters", NV / "dmir_cls_v4" / "model_1790356146757549162_1029274.h5", 152250),
    ("hand-designed network of July, 83,803 parameters", DEPLOY / "cls_best.h5", None),
    ("the lab's reference CNN, 441,347 parameters", DEPLOY / "REF_cnn_multi.h5", None),
)
GO_POINTS, GO_SE = 0.02, 2.0


def train(isc, h5: str, ds, seed: int):
    """input_scaling_check.train for a classifier; a softmax output layer is made linear first."""
    keras, sv, tf = isc.keras, isc.sv, isc.tf
    keras.utils.set_random_seed(seed)
    net = keras.models.clone_model(keras.models.load_model(h5, compile=False))
    last = net.layers[-1]
    if getattr(last, "activation", None) is keras.activations.softmax:
        last.activation = keras.activations.linear
    loss, metrics = sv.compile_args("logits")
    net.compile(optimizer="adam", loss=loss, metrics=metrics)
    tr = ds.train_dataset().shuffle(sv.BATCH * 8, seed=seed).batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    va = ds.validation_dataset().batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    hist = net.fit(tr, validation_data=va, epochs=sv.EPOCH_CAP or 80, verbose=0,
                   callbacks=sv.callbacks("dmir", "logits"))
    return net, len(hist.history["loss"])


def run(which):
    import input_clip_check as icc
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters

    tf, keras, sv = isc.tf, isc.keras, isc.sv
    os.environ["DMIR_SCALE"] = ""
    KEEP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = ({(r["candidate"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())}
            if OUT.exists() else set())
    smoke = "_smoke" if sv.EPOCH_CAP else ""
    table = cv.sessions_table()
    x, y, counts = stc.stream_windows(*cv.pack(table, [f"user{u}LC" for u in TRAIN]), "LC", np.random.default_rng(0))
    val_names = [f"user{u}{ph}" for u in VAL for ph in ("LC", "N")]
    zv, xv = cv.pack(table, val_names)
    val = stc.stream_windows(zv, xv, "LC", np.random.default_rng(0))[:2]
    xl, yl = sv.DMIR_Dataset(task=cv.TASK)._data["val"]      # the lab's validation set, for information
    rng = np.random.default_rng(42)
    calib = np.clip(x[rng.choice(len(x), size=min(500, len(x)), replace=False)], -stc.BOUND, stc.BOUND).astype(np.float32)
    for idx in which:
        name, h5, macs = CANDIDATES[idx]
        for seed in sv.SEEDS:
            if (name, seed) in done:
                continue
            t0 = time.perf_counter()
            kept = KEEP / f"c{idx}_seed{seed}{smoke}.h5"
            try:
                if kept.exists():
                    net, epochs = keras.models.load_model(str(kept), compile=False), None
                else:
                    net, epochs = train(isc, str(h5), stc.Windows(tf, (x, y), val), seed)
                    net.save(str(kept))
            except Exception as e:                           # a reference file that does not load is left out
                print(f"candidate {idx} ({name}) left out: {type(e).__name__}: {str(e)[:200]}", flush=True)
                break
            conv = tf.lite.TFLiteConverter.from_keras_model(net)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = lambda s=calib: ([s[i:i + 1]] for i in range(len(s)))
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
            tfl = conv.convert()
            (KEEP / f"c{idx}_seed{seed}{smoke}_int8_io.tflite").write_bytes(tfl)
            outs, last, scale = stc.drive_outputs(tf, icc, tfl, zv, xv)
            sessions, acc, after = stc.drive_measures(zv, outs, last, scale)
            rec = {"candidate": name, "index": idx, "seed": seed, "macs": macs, "params": int(net.count_params()),
                   "training_windows": int(len(x)), "scored_frames_per_class_before_balancing": counts,
                   "epochs_run": epochs, "int8_io_bytes": len(tfl),
                   "validation_drivers": {"frame_accuracy": acc, "output_after_a_lane_change": after,
                                          "raw_output": dr.score(sessions, dr.RAW)},
                   "lab_validation_set_accuracy_int8": round(isc.metric(icc.predict_int8(tf, tfl, xl)[0], yl, True)["acc"], 5),
                   "wall_s": round(time.perf_counter() - t0, 1),
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"candidate {idx} seed {seed}: {epochs} epochs, {rec['wall_s']} s", flush=True)
            keras.backend.clear_session()


def report():
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    table = []
    for name, _, macs in CANDIDATES:
        rs = sorted((r for r in rows if r["candidate"] == name), key=lambda r: r["seed"])
        if not rs:
            continue
        v = [r["validation_drivers"] for r in rs]
        lc = [x["frame_accuracy"]["lane_change_sessions"] for x in v]
        table.append({"candidate": name, "macs": macs, "params": rs[0]["params"], "seeds": [r["seed"] for r in rs],
                      "int8_io_bytes": rs[0]["int8_io_bytes"],
                      "frames_correct_sessions_with_lane_changes": {"mean": round(st.mean(lc), 4),
                                                                    "std": round(st.stdev(lc), 4) if len(lc) > 1 else None,
                                                                    "values": lc},
                      "frames_correct_all_sessions": round(st.mean(x["frame_accuracy"]["all"] for x in v), 4),
                      "frames_correct_baseline_sessions": round(st.mean(x["frame_accuracy"]["baseline_sessions"] for x in v), 4),
                      "opposite_announced_after_a_lane_change": round(st.mean(
                          x["output_after_a_lane_change"]["the opposite direction"] for x in v), 4),
                      "timely_share": round(st.mean(x["raw_output"]["timely_share"] for x in v), 4),
                      "alarm_runs_per_hour": round(st.mean(x["raw_output"]["false_alarms_per_hour"] for x in v), 1),
                      "alarm_runs_of_0.5_s_or_more_per_hour": round(st.mean(
                          x["raw_output"]["false_alarms_of_0.5_s_or_more_per_hour"] for x in v), 1),
                      "lab_validation_set_accuracy_int8": round(st.mean(r["lab_validation_set_accuracy_int8"] for r in rs), 4),
                      "epochs_run_mean": round(st.mean(r["epochs_run"] for r in rs if r["epochs_run"]), 1)
                      if any(r["epochs_run"] for r in rs) else None})
    key = "frames_correct_sessions_with_lane_changes"
    base = next((t for t in table if t["candidate"] == DEPLOYED), None)
    best = max(table, key=lambda t: t[key]["mean"])
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "records": len(rows),
           "measure": "share of correct frames, validation drivers' sessions with lane changes, int8, mean over seeds",
           "candidates": table}
    if base and base[key]["std"] is not None and best[key]["std"] is not None:
        gain = best[key]["mean"] - base[key]["mean"]
        se = (best[key]["std"] ** 2 / len(best["seeds"]) + base[key]["std"] ** 2 / len(base["seeds"])) ** 0.5
        go = best is not base and gain >= GO_POINTS - 1e-12 and gain >= GO_SE * se
        within = [t for t in table if t[key]["mean"] >= best[key]["mean"] - 0.01 and t["macs"]]
        res["rule"] = {"text": "GO if the best candidate is at least 2 points above the deployed network and the "
                               "difference is at least twice its standard error",
                       "deployed": base[key]["mean"], "best": best["candidate"], "best_value": best[key]["mean"],
                       "gain": round(gain, 4), "standard_error": round(se, 4), "go": bool(go),
                       "cheapest_within_one_point_of_the_best_for_information":
                           min(within, key=lambda t: t["macs"])["candidate"] if within else None}
    SUMMARY.write_text(json.dumps(res, indent=1))
    for t in table:
        m = t[key]
        print(f"{t['candidate'][:52]:52s} frames correct {m['mean']:.4f} +- {m['std']} (all sessions "
              f"{t['frames_correct_all_sessions']:.4f}); opposite {t['opposite_announced_after_a_lane_change']:.3f}; "
              f"timely {t['timely_share']:.3f}, {t['alarm_runs_per_hour']}/h; lab val set "
              f"{t['lab_validation_set_accuracy_int8']:.4f}; {len(t['seeds'])} seeds")
    print(json.dumps(res.get("rule"), indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        run([int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(len(CANDIDATES))))
