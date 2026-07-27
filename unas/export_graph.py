"""Export a searched Keras-3 model as an adjacency graph + per-layer weights.

Why this exists: the searched models are saved by the fork under **Keras 3**,
but tfmot (needed for QAT) only works under **tf_keras** (Keras 2), and the two
serialize functional graphs incompatibly (`batch_shape` vs `batch_input_shape`,
different `inbound_nodes` layout). Rather than fight the JSON, we export what we
actually need — the layer list, their configs, and the wiring — and rebuild the
model natively on the other side (`unas/qat_finetune.py`).

Run under the Keras-3 env (the fork's default):
  source ~/dmir_nas/env.sh
  ~/dmir_nas/bin/python unas/export_graph.py <model.h5> <out_dir>

Writes <out_dir>/<stem>_graph.json and <out_dir>/<stem>_weights.pkl.
"""
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras


def inbound_names(layer):
    """Names of the layers feeding this one, in call order (Keras 3 internals)."""
    names = []
    for node in layer._inbound_nodes:
        tensors = node.input_tensors
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        for t in tensors:
            hist = getattr(t, "_keras_history", None)
            if hist is None:
                continue
            op = hist[0] if isinstance(hist, tuple) else hist.operation
            names.append(op.name if hasattr(op, "name") else str(op))
    return names


def main(h5_path, out_dir):
    m = keras.models.load_model(h5_path, compile=False)
    stem = Path(h5_path).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    outputs = m.outputs if isinstance(m.outputs, (list, tuple)) else [m.outputs]
    graph = {
        "source": str(h5_path),
        "params": int(m.count_params()),
        "input_names": [l.name for l in m.layers
                        if l.__class__.__name__ == "InputLayer"],
        "output_names": [t._keras_history.operation.name for t in outputs],
        "layers": [{"name": l.name,
                    "class_name": l.__class__.__name__,
                    "config": l.get_config(),
                    "inputs": inbound_names(l)} for l in m.layers],
    }
    (out / f"{stem}_graph.json").write_text(json.dumps(graph, indent=1))
    with open(out / f"{stem}_weights.pkl", "wb") as f:
        pickle.dump({l.name: l.get_weights() for l in m.layers if l.get_weights()}, f)

    print(f"{stem}: {len(m.layers)} layers, {m.count_params():,} params")
    print(f"  classes: {sorted({l['class_name'] for l in graph['layers']})}")
    print(f"  -> {out / f'{stem}_graph.json'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
