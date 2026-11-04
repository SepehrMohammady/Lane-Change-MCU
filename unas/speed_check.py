"""Does Keras steps_per_execution change the training, and how much faster is it? (2026-11-03)

The searches train networks of a few thousand parameters; one training step takes about 2 ms, almost
all of it per-step dispatch, and the GPU sits at 11-17 % during the Part A pilot. steps_per_execution=N
runs N batches per call. This trains one saved network with the select_by_seeds.train_one recipe
(Adam, batch 256, seeded shuffle) for a few epochs per setting, outside Ray:
  1. timing: seconds per epoch (training + validation) for each N
  2. equivalence, with TF op determinism on: per-epoch loss and validation accuracy for each N
     against N=1, same seed

Run in the WSL dmir_nas venv, from the fork:
  cd ~/uNAS && HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/speed_check.py [epochs] [N,N,...]
Default network: the final highD classifier search's choice. Prints only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_by_seeds as sel                               # noqa: E402  (sets up TF, keras, adapters)

tf, keras, sv = sel.tf, sel.keras, sel.sv
H5 = sel.REPO / "datasets/highd/results/nas-v2/highd_cls_v4/model_1790282740507531593_621856.h5"
SEED = 100


class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_s"] = time.perf_counter() - self.t0


def fit(ds, spe: int, epochs: int) -> dict:
    keras.utils.set_random_seed(SEED)
    net = keras.models.clone_model(keras.models.load_model(H5, compile=False))
    loss, metrics = sv.compile_args("logits")
    net.compile(optimizer="adam", loss=loss, metrics=metrics, steps_per_execution=spe)
    train = ds.train_dataset().shuffle(sv.BATCH * 8, seed=SEED).batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    val = ds.validation_dataset().batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    hist = net.fit(train, validation_data=val, epochs=epochs, verbose=0, callbacks=[EpochTimer()])
    keras.backend.clear_session()
    return hist.history


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    spes = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 8, 32, 128]
    ds = sel.sv.HighD_Dataset(task="highd_cls")
    acc = next(k for k in fit(ds, 1, 1) if k.startswith("val_") and "acc" in k)   # warm-up, metric name

    print(f"timing, {epochs} epochs each (first epoch includes tracing):", flush=True)
    base = None
    for spe in spes:
        h = fit(ds, spe, epochs)
        later = sorted(h["epoch_s"][1:]) or h["epoch_s"]
        s = later[len(later) // 2]
        base = base or s
        print(f"  N={spe:4d}: first epoch {h['epoch_s'][0]:.1f} s, median later epoch {s:.2f} s "
              f"({base / s:.1f}x)", flush=True)

    tf.config.experimental.enable_op_determinism()
    print(f"equivalence with op determinism, {epochs} epochs, seed {SEED}:", flush=True)
    ref = fit(ds, 1, epochs)
    for spe in spes:
        h = ref if spe == 1 else fit(ds, spe, epochs)
        dl = max(abs(a - b) for a, b in zip(h["loss"], ref["loss"]))
        da = max(abs(a - b) for a, b in zip(h[acc], ref[acc]))
        print(f"  N={spe:4d}: {acc} " + " ".join(f"{v:.4f}" for v in h[acc])
              + f" | max diff to N=1: loss {dl:.2e}, {acc} {da:.2e}", flush=True)


if __name__ == "__main__":
    main()
