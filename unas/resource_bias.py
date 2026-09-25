"""How far the fork's resource model is from the networks the search actually trains.

The ELIOS fork builds every 1-D candidate twice: as a Keras model, with "same" padding in
each convolution, and as a resource graph for the search's bounds and objectives, with
"valid" padding (uNAS/cnn1d/cnn1d_architecture.py). Without padding every k-wide
convolution shortens the sequence by k-1 steps and later kernels are clipped to the
shorter length, so the graph under-counts MACs, weights and activations; the shorter the
input window, the larger the gap. The 2-D code the port started from uses "valid" in both.
A smaller second mismatch: the optional max-pool before a convolution rounds the length
down in Keras (padding "valid") and up in the graph.

For every evaluated point of every search this script compares the fork's stored
resource features with (a) MACs and weight count taken layer by layer from the Keras
model (the network that is trained, retrained and deployed) and (b) the resource graph
of FaithfulGapCnn1DArchitecture (unas/cnn1d_gap.py), which should give exactly (a) and
also gives peak memory. Counting follows the fork's conventions: MACs include pooling and
additions, model size counts kernel and bias elements (one byte each), batch
normalization is folded. Candidates whose training failed (the fork stores 1e12 for
their features) are listed but left out of the ratios.

    cd ~/uNAS && CUDA_VISIBLE_DEVICES= ~/dmir_nas/bin/python <repo>/unas/resource_bias.py [search ...]

Writes datasets/highd/results/resource_bias.json (per search: n, median and range of the
true/stored ratios, and every point's numbers; under "hand_like_in_v2_space" the three counts
of the hand-designed layer sequence, which set the v4 budgets). --hand-only refreshes only those.
"""
import contextlib
import importlib.util
import io
import json
import os
import pickle
import statistics
import sys
from pathlib import Path

HOME = Path.home() / "uNAS"
sys.path[:0] = [str(HOME), str(HOME / "configs")]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
from uNAS.resource_models.models import macs, model_size, peak_memory_usage  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ART = HOME / "artifacts"
_spec = importlib.util.spec_from_file_location("cnn1d_gap_repo", REPO / "unas" / "cnn1d_gap.py")
gap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gap)

# search -> (input shape, number of classes)
SEARCHES = {
    "dmir_cls": ((50, 31), 3), "dmir_cls_noind": ((50, 29), 3),
    "dmir_lcr": ((50, 31), 1), "dmir_lcl": ((50, 31), 1),
    "dmir_lcr_rmse": ((50, 31), 1), "dmir_lcl_rmse": ((50, 31), 1),
    "highd_cls": ((10, 18), 3), "highd_cls_tight": ((10, 18), 3),
    "highd_cls_v2": ((10, 18), 3), "highd_cls_v3": ((10, 18), 3),
    "highd_ttlc": ((10, 18), 1), "highd_ttlc_v2_p": ((10, 18), 1),
    "highd_cls_v4": ((10, 18), 3), "highd_ttlc_v4": ((10, 18), 1),
}
FAILED = 10 ** 12


def faithful_features(arch_dict, shape, n_cls):
    g = gap.FaithfulGapCnn1DArchitecture(arch_dict).to_resource_graph(shape, n_cls)
    return [int(peak_memory_usage(g)), int(model_size(g)), int(macs(g))]


def keras_features(model):
    """Weight elements and MACs of the Keras model, counted the way the fork counts them."""
    n_macs, n_w = 0, 0
    for layer in model.layers:
        kind = type(layer).__name__
        out = layer.output.shape
        if kind in ("Conv1D", "DepthwiseConv1D", "Dense"):
            w = layer.get_weights()
            n_w += sum(int(x.size) for x in w)
            k = w[0]
            if kind == "Dense":
                n_macs += k.shape[0] * k.shape[1]
            else:   # Conv1D kernel (k, in, out); DepthwiseConv1D kernel (k, channels, 1)
                n_macs += out[1] * k.shape[0] * k.shape[1] * k.shape[2]
        elif kind in ("MaxPooling1D", "AveragePooling1D"):
            n_macs += out[1] * out[2] * layer.pool_size[0]
        elif kind == "GlobalAveragePooling1D":
            n_macs += layer.input.shape[1] * layer.input.shape[2]
        elif kind == "Add":
            n_macs += (len(layer.input) - 1) * out[1] * out[2]
    return n_w, int(n_macs)


def ratio_stats(values):
    return {"median": round(statistics.median(values), 3), "min": round(min(values), 3),
            "max": round(max(values), 3)}


def run(search):
    # renamed runs (HIGHD_V2_SUFFIX) take the shape of the search they extend
    shape, n_cls = SEARCHES.get(search) or next(v for k, v in SEARCHES.items() if search.startswith(k))
    with open(ART / search / f"{search}_agingevosearch_state.pickle", "rb") as f:
        history = pickle.load(f)
    rows = []
    for i, p in enumerate(history):
        arch = p.point.arch
        with contextlib.redirect_stdout(io.StringIO()):
            model = arch.to_keras_model(shape, n_cls)
        n_w, n_macs = keras_features(model)
        blocks = arch.architecture["conv_blocks"]
        rows.append({"i": i, "val_error": float(p.val_error),
                     "stored": [int(v) for v in p.resource_features],
                     "keras": {"weights": n_w, "macs": n_macs},
                     "faithful": faithful_features(arch.architecture, shape, n_cls),
                     "prepool": any(l.get("has_prepool") for b in blocks for l in b["layers"]),
                     "branch": any(b.get("is_branch") for b in blocks)})
        if i % 50 == 49:
            keras.backend.clear_session()
    ok = [r for r in rows if r["stored"][0] < FAILED]
    out = {
        "search": search, "input_shape": list(shape), "n": len(rows), "failed": len(rows) - len(ok),
        "macs_ratio": ratio_stats([r["keras"]["macs"] / r["stored"][2] for r in ok]),
        "size_ratio": ratio_stats([r["keras"]["weights"] / r["stored"][1] for r in ok]),
        "pmu_ratio": ratio_stats([r["faithful"][0] / r["stored"][0] for r in ok]),
        "faithful_equals_keras": sum(r["faithful"][2] == r["keras"]["macs"]
                                     and r["faithful"][1] == r["keras"]["weights"] for r in rows),
        "points": rows,
    }
    m, s, p = out["macs_ratio"], out["size_ratio"], out["pmu_ratio"]
    print(f"{search:16s} n={out['n']:4d} failed={out['failed']:2d}  MACs x{m['median']} ({m['min']}-{m['max']})  "
          f"size x{s['median']} ({s['min']}-{s['max']})  PMU x{p['median']} ({p['min']}-{p['max']})  "
          f"faithful graph = Keras: {out['faithful_equals_keras']}/{out['n']}", flush=True)
    return out


def hand_like_costs():
    """The hand-designed layer sequence in the v2 space (unas/build_hand_like.py): the fork's
    count, the faithful count and the Keras count; the v4 budgets are its faithful count."""
    spec = importlib.util.spec_from_file_location("build_hand_like_repo", REPO / "unas" / "build_hand_like.py")
    hand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hand)
    out = {}
    for task, n_out in (("highd_cls", 3), ("highd_ttlc", 1)):
        g = gap.GapCnn1DArchitecture(hand.hand_like()).to_resource_graph((10, 18), n_out)
        with contextlib.redirect_stdout(io.StringIO()):
            model = gap.GapCnn1DArchitecture(hand.hand_like()).to_keras_model((10, 18), n_out)
        n_w, n_macs = keras_features(model)
        out[task] = {"fork": [int(peak_memory_usage(g)), int(model_size(g)), int(macs(g))],
                     "faithful": faithful_features(hand.hand_like(), (10, 18), n_out),
                     "keras": {"weights": n_w, "macs": n_macs}}
        print(f"hand-designed layer sequence, {task}: {out[task]}")
    return out


if __name__ == "__main__":
    # --hand-only: refresh only the hand-designed layer sequence's counts
    names = [a for a in sys.argv[1:] if a != "--hand-only"]
    results = {"hand_like_in_v2_space": hand_like_costs()}
    if "--hand-only" in sys.argv:
        names = []
    elif not names:
        names = list(SEARCHES)
    for name in names:
        if not (ART / name / f"{name}_agingevosearch_state.pickle").exists():
            print(f"{name}: no search state, skipped")
            continue
        results[name] = run(name)
    dest = REPO / "datasets" / "highd" / "results" / "resource_bias.json"
    old = json.loads(dest.read_text()) if dest.exists() else {}
    old.update(results)
    dest.write_text(json.dumps(old, indent=1) + "\n")
    print("written", dest)
