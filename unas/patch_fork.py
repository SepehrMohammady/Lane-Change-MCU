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


def main(path: str) -> None:
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
