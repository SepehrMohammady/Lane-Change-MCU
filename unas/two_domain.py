"""Per-domain validation for the two-domain search pilot (highD + exiD pooled, 2026-11-04).

DomainValidation scores the model on each domain's validation windows after every epoch and adds
val_acc_<domain> and val_worst_accuracy (the lower of the two) to the epoch logs. Keras passes one logs
dict through the callbacks in list order and appends History last, so EarlyStopping and the training
history see these keys when DomainValidation comes first.

The two arms of the pilot differ only in the key they read (OBJECTIVES):
  pooled  val_accuracy, the fork's objective on the pooled validation windows
  worst   val_worst_accuracy, for early stopping and as the search's validation error
          (NAS_VAL_KEY in the fork's trainer, unas/patch_fork.py patch 4)
Both arms run DomainValidation, so both spend the same compute per training run.

Used by the fork config (unas/highd_config.py, copied to configs/two_domain.py) and by
unas/select_by_seeds.py and unas/final_eval.py.
"""
import keras
import numpy as np

OBJECTIVES = {"pooled": "val_accuracy", "worst": "val_worst_accuracy"}


class DomainValidation(keras.callbacks.Callback):
    def __init__(self, domains, batch_size=4096):
        super().__init__()
        self.val = {d: splits["val"] for d, splits in domains.items()}
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        accs = []
        for d, (x, y) in self.val.items():
            p = self.model.predict(x, batch_size=self.batch_size, verbose=0)
            accs.append(float(np.mean(p.argmax(1) == y)))
            logs[f"val_acc_{d}"] = accs[-1]
        logs["val_worst_accuracy"] = min(accs)


def callbacks(domains, objective):
    """The highD classifier recipe of the searches (unas/highd_config.py, unas/seed_variance.py)
    with per-domain validation in front and early stopping on the arm's objective."""
    return [DomainValidation(domains),
            keras.callbacks.EarlyStopping(monitor=OBJECTIVES[objective], mode="max", patience=8,
                                          restore_best_weights=True),
            keras.callbacks.TerminateOnNaN()]
