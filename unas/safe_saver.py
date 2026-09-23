"""Model saver for the ELIOS uNAS fork that survives chunked runs and parallel workers.

Copied into the fork as configs/safe_saver.py by unas/run_chunked_highd.sh; the *_v2
search configs call install(), which makes the fork build this saver instead of its own.

The fork's ModelSaver restarts its file counter in every process, and every Ray worker
holds its own copy, so chunked or parallel searches write the same file names
(model_aaaaaa.h5, ...) into one folder and overwrite each other; its metadata then no
longer describes the files on disk. It also applies the error bound and the Pareto test
to the test error. This saver:

  * gives every file a unique name (time in ns and process id);
  * writes a JSON sidecar next to each .h5: validation error, resource features,
    parameter count, layer configs, and the test error that the fork computes (kept for
    the record, never used for any decision);
  * keeps every candidate within the resource bounds (peak memory, model size, MACs)
    and deletes nothing: error-based selection happens afterwards, on validation data
    only (unas/select_by_seeds.py);
  * writes no shared state during the search; save_models() gathers the sidecars into
    metadatas.json at the end of each chunk.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from uNAS.model_saver import ModelSaver
from uNAS.utils import NumpyEncoder


class SafeModelSaver(ModelSaver):
    def _within_resource_bounds(self, obj) -> bool:
        return all(b is None or obj[k] <= b for k, b in
                   (("pmu", self.peak_mem_bound), ("ms", self.model_size_bound), ("macs", self.mac_bound)))

    def evaluate_and_save(self, model, val_error, test_error, resource_features):
        if self.save_criteria == "none":
            return
        obj = self._pack_model(model, val_error, test_error, resource_features)
        if self.save_criteria != "all" and not self._within_resource_bounds(obj):
            print(f"Not saved: outside the resource bounds (pmu {obj['pmu']}, ms {obj['ms']}, macs {obj['macs']})")
            return
        keras_model = obj.pop("model")
        name = f"model_{time.time_ns()}_{os.getpid()}"
        keras_model.save(os.path.join(self.models_path, name + ".h5"))
        obj.update(model_name=name + ".h5", params=int(keras_model.count_params()),
                   saved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with open(os.path.join(self.models_path, name + ".json"), "w") as f:
            json.dump(obj, f, cls=NumpyEncoder)
        print(f"Saved {name}.h5 (val_error {float(val_error):.4f})")

    def save_models(self):
        rows = [json.loads(p.read_text()) for p in sorted(Path(self.models_path).glob("model_*.json"))]
        with open(os.path.join(self.full_path, "metadatas.json"), "w") as f:
            json.dump(rows, f, cls=NumpyEncoder, indent=1)
        print(f"{len(rows)} saved models described in {self.full_path}/metadatas.json")
        return rows


def install():
    """Make the fork construct SafeModelSaver.

    The module uNAS/uNAS.py binds ModelSaver at import and builds the saver in
    _configure_search_algorithm(), after the config module has been loaded. The
    package re-exports the class uNAS under the same name as that module, so the
    module is fetched with importlib (an attribute lookup would return the class).
    """
    import importlib
    core = importlib.import_module("uNAS.uNAS")
    core.ModelSaver = SafeModelSaver
