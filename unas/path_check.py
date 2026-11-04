"""Do the search's trainer and select_by_seeds.train_one score one network alike? (2026-11-03)

The Part A pilot found the search's own runs 0.5-1.1 points below final_eval.py for the same
architectures (paper/NOTES.md, 2026-11-03), although both use Adam, batch 256, 50 epochs and the
same callbacks. This trains one saved network with both code paths and the same seeds, outside Ray:
  A  uNAS/model_trainer.py ModelTrainer.train_and_eval (val_error = 1 - max val_accuracy)
  B  select_by_seeds.train_one (validation accuracy of the restored model)

Run in the WSL dmir_nas venv, from the fork:
  cd ~/uNAS && ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/path_check.py [h5] [seeds]
Default network: the final highD classifier search's choice (3,968 parameters). Prints only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_by_seeds as sel                               # noqa: E402  (sets up TF, keras, adapters)

from configs.highd_config import _training_config          # noqa: E402  (the fork's search config)
from uNAS.model_trainer import ModelTrainer                 # noqa: E402

keras = sel.keras
H5 = sel.REPO / "datasets/highd/results/nas-v2/highd_cls_v4/model_1790282740507531593_621856.h5"


def main() -> None:
    h5 = Path(sys.argv[1]) if len(sys.argv) > 1 else H5
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [100, 101, 102]
    ds = sel.sv.HighD_Dataset(task="highd_cls")
    trainer = ModelTrainer(_training_config(ds, True))
    for seed in seeds:
        keras.utils.set_random_seed(seed)
        net = keras.models.clone_model(keras.models.load_model(h5, compile=False))
        a = trainer.train_and_eval(net)
        keras.backend.clear_session()
        b = sel.train_one(str(h5), ds, "cls", seed)
        print(f"seed {seed}: A search trainer val {1 - a['val_error']:.4f} test {1 - a['test_error']:.4f} | "
              f"B train_one val {b['val']['acc']:.4f} test {b['test']['acc']:.4f} ({b['epochs_run']} epochs)",
              flush=True)


if __name__ == "__main__":
    main()
