"""Does the board cost depend on the weight values? A controlled check.

Builds the hand-designed highD classifier twice from one bundle of
scripts/export_hand_cnn.py: once with its trained weights and once with random
weights (same graph, same shapes, seeded), converts both exactly as
unas/deploy_hand_cnn.py does (float32, and full int8 with int8 I/O calibrated on the
same 500 windows) and writes the four files for unas/st_benchmark.py. If latency,
flash and RAM agree within the farm's repeatability, cost measurements of models whose
trained weights we do not have (random weights, same graph) stand.

Run in the WSL dmir_nas venv:
  source ~/dmir_nas/env.sh
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/weight_check.py <bundle.npz> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy_hand_cnn as d                               # noqa: E402  (sets up TF / Keras)


def randomize(model, seed=0):
    rng = np.random.default_rng(seed)
    for layer in model.layers:
        ws = layer.get_weights()
        if not ws:
            continue
        new = []
        for w in ws:
            r = rng.normal(0.0, 0.1, w.shape).astype(w.dtype)
            if "BatchNormalization" in layer.__class__.__name__ and len(new) == 3:
                r = np.abs(r) + 0.5                          # moving variance must stay positive
            new.append(r)
        layer.set_weights(new)
    return model


def main(bundle, out_dir):
    z = np.load(bundle)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(bundle).stem
    calib = z["x_calib"].astype(np.float32)
    for tag, model in (("trained", d.build(z)), ("random", randomize(d.build(z)))):
        for name, mode, io in (("f32", "f32", False), ("int8_io", "int8", True)):
            tfl = d.convert(model, calib, mode, io)
            path = out / f"{stem}_{tag}_{name}.tflite"
            path.write_bytes(tfl)
            print(f"{path.name}: {len(tfl)} B")


if __name__ == "__main__":
    main(*sys.argv[1:])
