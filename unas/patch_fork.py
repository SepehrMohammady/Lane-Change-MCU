"""Idempotent patches for the ELIOS uNAS fork (aging_evolution.py).

1. Session cleanup: free the TF graph after each candidate — the persistent
   ray GPUTrainer actor otherwise leaks ~200 MB/candidate and the search dies
   on a host-RAM OOM after ~40 candidates.
2. Safe evaluate: one failing candidate (TF Graph-execution error, VRAM
   blow-up on an oversized architecture, ...) must not kill the whole search.
   Catch, clean up, and return a worst-score point so evolution discards it.

Usage:  python3 patch_fork.py /path/to/uNAS/uNAS/search_algorithms/aging_evolution.py
"""
import sys
from pathlib import Path

GUARD = '''
    def _safe_evaluate(self, point):
        try:
            return self._orig_evaluate(point)
        except Exception as e:  # noqa: BLE001 - any candidate failure
            import gc
            import logging
            logging.getLogger("Worker").error(
                f"Candidate failed ({type(e).__name__}); scoring as worst. {str(e)[:200]}")
            try:
                import keras
                keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()
            big = 10 ** 12
            return EvaluatedPoint(point=point, val_error=1.0, test_error=1.0,
                                  resource_features=[big, big, big])

    evaluate = _safe_evaluate
'''

CLEANUP_ANCHOR_1 = (
    "        if self.model_saver:\n"
    "            self.model_saver.evaluate_and_save(model, val_error, test_error, resource_features)\n"
)
CLEANUP = (
    "\n        del model, rg\n"
    "        keras.backend.clear_session()\n"
    "        gc.collect()\n"
)

RETURN_1 = (
    "        return EvaluatedPoint(point=point,\n"
    "                              val_error=val_error, test_error=test_error,\n"
    "                              resource_features=resource_features)\n"
)
RETURN_2 = (
    "        return EvaluatedPoint(\n"
    "            point=point,\n"
    "            val_error=val_error, \n"
    "            test_error=test_error,\n"
    "            resource_features=resource_features\n"
    "        )\n"
)


REG_LOSS_ANCHOR = (
    "        if dataset.num_classes < 2:#Regression\n"
    "            loss = tf.keras.losses.MeanAbsoluteError() \n"
    "            metric = tf.keras.metrics.MeanAbsoluteError(name=\"mae\")\n"
)
REG_LOSS_NEW = (
    "        if dataset.num_classes < 2:#Regression\n"
    "            import os as _os\n"
    "            if _os.environ.get('DMIR_REG_METRIC', 'mae') == 'rmse':\n"
    "                loss = tf.keras.losses.MeanSquaredError()\n"
    "                metric = tf.keras.metrics.RootMeanSquaredError(name='rmse')\n"
    "            else:\n"
    "                loss = tf.keras.losses.MeanAbsoluteError()\n"
    "                metric = tf.keras.metrics.MeanAbsoluteError(name='mae')\n"
)
REG_VALERR_ANCHOR = (
    "            val_error = log.history[\"val_mae\"][-1] if self.pruning "
    "and self.pruning.finish_pruning_by_epoch >= epochs else "
    "min(log.history[\"val_mae\"][check_logs_from_epoch:])\n"
)
REG_VALERR_NEW = (
    "            import os as _os\n"
    "            _mk = 'val_rmse' if _os.environ.get('DMIR_REG_METRIC', 'mae') == 'rmse' else 'val_mae'\n"
    "            val_error = log.history[_mk][-1] if self.pruning "
    "and self.pruning.finish_pruning_by_epoch >= epochs else "
    "min(log.history[_mk][check_logs_from_epoch:])\n"
)


def patch_trainer(fork_root: str) -> None:
    """Make regression RMSE-aware when DMIR_REG_METRIC=rmse (model_trainer.py)."""
    p = Path(fork_root) / "uNAS" / "model_trainer.py"
    s = p.read_text(encoding="utf-8")
    if "DMIR_REG_METRIC" in s:
        print("model_trainer already RMSE-aware"); return
    assert REG_LOSS_ANCHOR in s, "regression loss anchor not found"
    assert REG_VALERR_ANCHOR in s, "regression val_error anchor not found"
    s = s.replace(REG_LOSS_ANCHOR, REG_LOSS_NEW)
    s = s.replace(REG_VALERR_ANCHOR, REG_VALERR_NEW)
    p.write_text(s, encoding="utf-8")
    print("model_trainer patched: RMSE-aware regression")


def main(path: str) -> None:
    # path is .../uNAS/search_algorithms/aging_evolution.py; the fork root is
    # three levels up. Also patch the sibling model_trainer.py.
    fork_root = Path(path).resolve().parents[2]
    patch_trainer(str(fork_root))

    with open(path, encoding="utf-8") as f:
        s = f.read()

    if "import gc" not in s:
        s = s.replace("import ray\n", "import ray\nimport gc\nimport keras\n", 1)

    if "keras.backend.clear_session()" not in s:
        s = s.replace(CLEANUP_ANCHOR_1, CLEANUP_ANCHOR_1 + CLEANUP)
    print("session-cleanup sites:", s.count("keras.backend.clear_session()"))

    if "_safe_evaluate" not in s:
        s = s.replace("    def evaluate(self, point):",
                      "    def _orig_evaluate(self, point):")
        assert s.count("def _orig_evaluate") == 2, "expected two actor classes"
        s = s.replace(RETURN_1, RETURN_1 + GUARD, 1)
        s = s.replace(RETURN_2, RETURN_2 + GUARD, 1)
        assert s.count("_safe_evaluate") >= 4, "guard not inserted twice"
    print("safe-evaluate wrappers:", s.count("evaluate = _safe_evaluate"))

    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print("patched OK:", path)


if __name__ == "__main__":
    main(sys.argv[1])
