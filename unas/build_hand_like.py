"""The hand-designed DSCNN's layer sequence, expressed in the v2 search space.

Builds, with configs/cnn1d_gap.py (the fork's 1D space plus the global-average-pooling
head), the architecture closest to the hand-designed CNN: a stride-2 Conv1D stem with 32
filters, two blocks of stride-2 depthwise Conv1D + 1x1 Conv1D (48, 64 filters) with
BatchNorm and ReLU, global average pooling, dropout 0.2, output layer. It differs from the
PyTorch network only in what the space fixes: convolution biases and "same" padding.
Trained with seed_variance.py (key highd_cls_hand_in_v2), it separates the search from
the training recipe: same architecture as the hand-designed CNN, recipe of the search.

Run in the WSL dmir_nas venv, from the fork (after run_chunked_highd.sh has copied
cnn1d_gap.py into configs/):
  source ~/dmir_nas/env.sh; cd ~/uNAS
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/build_hand_like.py <out.h5> [n_features] [n_outputs]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.expanduser("~/uNAS"))
from configs.cnn1d_gap import GapCnn1DArchitecture   # noqa: E402


def layer(kind, **kw):
    return {"type": kind, "has_bn": False, "has_relu": False, "has_prepool": False, **kw}


def hand_like() -> dict:
    stride2 = {"1x_stride": True}          # the fork's flag name: True selects stride 2
    bn_relu = {"has_bn": True, "has_relu": True}
    return {
        "conv_blocks": [
            {"is_branch": False, "layers": [layer("Conv1D", ker_size=5, filters=32, **stride2) | bn_relu]},
            {"is_branch": False, "layers": [layer("DWConv1D", ker_size=5, **stride2),
                                            layer("1x1Conv1D", filters=48) | bn_relu]},
            {"is_branch": False, "layers": [layer("DWConv1D", ker_size=5, **stride2),
                                            layer("1x1Conv1D", filters=64) | bn_relu]},
        ],
        "pooling": {"type": "gap", "pool_size": 2},
        "dense_blocks": [],
        "head_dropout": 0.2,
    }


if __name__ == "__main__":
    out = sys.argv[1]
    n_features = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    n_outputs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    model = GapCnn1DArchitecture(hand_like()).to_keras_model((10, n_features), n_outputs)
    model.save(out)
    print(f"{out}: {model.count_params()} parameters (BatchNorm statistics included)")
