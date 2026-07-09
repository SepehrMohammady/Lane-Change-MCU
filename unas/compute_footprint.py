"""Estimate MCU footprint of a deployment model directly from its architecture.

Gives paper-ready provisional numbers before ST Edge AI confirms them:
  * flash   ~ the int16x8 .tflite size (weights int8 + runtime metadata);
  * MACs    = multiply-accumulate ops per inference (Conv1D + Dense);
  * peak RAM ~ max over layers of (input + output) activation bytes, the
    one-operator-at-a-time working set (int16 activations => 2 bytes/elem),
    the same notion muNAS optimises. This is an estimate; ST Edge AI reports
    the exact peak SRAM and adds measured latency.

  ~/dmir_nas/bin/python compute_footprint.py <h5_path> [tflite_path]
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

BYTES_PER_ACT = 2  # int16 activations


def layer_io_elems(layer):
    def n(shape):
        return int(np.prod([d for d in shape[1:] if d is not None])) if shape else 0
    try:
        return n(layer.input.shape), n(layer.output.shape)
    except Exception:
        return 0, 0


def macs_of(layer):
    cls = layer.__class__.__name__
    o = layer.output.shape
    if cls == "Conv1D":
        out_len, out_ch = o[1], o[2]
        k = layer.kernel_size[0]
        in_ch = layer.input.shape[-1]
        groups = getattr(layer, "groups", 1) or 1
        return int(out_len * out_ch * k * (in_ch // groups))
    if cls in ("DepthwiseConv1D",):
        out_len, ch = o[1], o[2]
        k = layer.kernel_size[0]
        return int(out_len * ch * k)
    if cls == "Dense":
        return int(layer.input.shape[-1] * o[-1])
    return 0


def main(h5, tflite=None):
    m = tf.keras.models.load_model(h5, compile=False)
    macs, peak = 0, 0
    for lyr in m.layers:
        macs += macs_of(lyr)
        i, o = layer_io_elems(lyr)
        peak = max(peak, (i + o) * BYTES_PER_ACT)
    flash_kb = Path(tflite).stat().st_size / 1024 if tflite and Path(tflite).exists() else None
    print(f"{Path(h5).stem:16} params={m.count_params():>7}  "
          f"MACs={macs:>10,}  peakRAM~{peak/1024:6.1f}KB  "
          f"flash(tflite)={flash_kb:6.1f}KB" if flash_kb else
          f"{Path(h5).stem:16} params={m.count_params():>7}  MACs={macs:>10,}  peakRAM~{peak/1024:.1f}KB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
