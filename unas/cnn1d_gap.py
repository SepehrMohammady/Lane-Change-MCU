"""1D search space of the ELIOS uNAS fork, extended with a global-average-pooling head.

Copied into the fork as configs/cnn1d_gap.py by unas/run_chunked_highd.sh and used by
the *_v2 search configs. The fork's head is always: optional pooling, Flatten, one to
three hidden Dense layers, output layer. It therefore cannot express the hand-designed
DSCNN (global average pooling, dropout, one linear layer), which matched or beat every
searched model in the seed study. This module adds, without editing the fork:

  * a global-average-pooling head, encoded as pooling type "gap";
  * zero to three hidden Dense layers (the fork allows one to three);
  * a per-architecture dropout rate (0, 0.1, 0.2 or 0.3) applied before each Dense
    layer, the way the fork applies its global dropout setting.

Convolution blocks, their mutations and the resource model (peak memory, model size,
MACs) are the fork's. Global average pooling enters the resource model as average
pooling over the full sequence length, which has the same output shape and cost.
With these options the space contains the hand-designed DSCNN's layer sequence.
"""
from copy import deepcopy

import numpy as np

from uNAS.cnn1d import Cnn1DArchitecture, Cnn1DSearchSpace
from uNAS.cnn1d.cnn1d_morphisms import produce_all_morphs
from uNAS.cnn1d.cnn1d_random_generators import random_arch_1d, random_dense_block
from uNAS.cnn1d.cnn1d_schema import get_schema
from uNAS.schema_types import Boolean, Categorical, Discrete

DROPOUTS = [0.0, 0.1, 0.2, 0.3]
MAX_DENSE = 3
GAP_PROB = 0.5
# pool_size is never read for "gap"; it is kept so the fork's pooling code finds the key
GAP = {"type": "gap", "pool_size": 2}


def is_gap(arch: dict) -> bool:
    p = arch.get("pooling")
    return bool(p) and p.get("type") == "gap"


class GapCnn1DArchitecture(Cnn1DArchitecture):
    _building = "keras"

    def _assemble_a_network(self, input, num_classes, conv_layer, pooling_layer,
                            dense_layer, add_layer, flatten_layer):
        def pooling(x, layer):
            if layer.get("type") != "gap":
                return pooling_layer(x, layer)
            if self._building == "keras":
                from keras.layers import GlobalAveragePooling1D
                return GlobalAveragePooling1D()(x)
            from uNAS.resource_models.ops import Pool1D
            return Pool1D(pool_size=x.shape[1], type="avg")(x)
        return super()._assemble_a_network(input, num_classes, conv_layer, pooling,
                                           dense_layer, add_layer, flatten_layer)

    def to_keras_model(self, input_shape, num_classes, dropout=0.0, **kwargs):
        self._building = "keras"
        return super().to_keras_model(input_shape, num_classes,
                                      dropout=self.architecture.get("head_dropout", dropout), **kwargs)

    def to_resource_graph(self, *args, **kwargs):
        self._building = "graph"
        try:
            return super().to_resource_graph(*args, **kwargs)
        finally:
            self._building = "keras"


class GapCnn1DSearchSpace(Cnn1DSearchSpace):
    """The fork's Cnn1DSearchSpace with the head options above."""

    @property
    def schema(self):
        s = dict(get_schema())
        s["num-dense-blocks"] = Discrete("num-dense-blocks", bounds=(0, MAX_DENSE))
        s["head-is-gap"] = Boolean("head-is-gap")
        s["head-dropout"] = Categorical("head-dropout", values=DROPOUTS)
        return s

    def random_architecture(self):
        arch = random_arch_1d().architecture
        if np.random.random_sample() < GAP_PROB:
            arch["pooling"] = dict(GAP)
        n_dense = np.random.randint(0, MAX_DENSE + 1)
        arch["dense_blocks"] = [random_dense_block(i) for i in range(n_dense)]
        arch["head_dropout"] = float(np.random.choice(DROPOUTS))
        return GapCnn1DArchitecture(arch)

    def produce_morphs(self, arch):
        parent = arch.architecture
        gap = is_gap(parent)
        # The fork's morphisms, applied with the GAP head hidden from them (they would
        # otherwise treat it as ordinary pooling and change its type or size).
        base = deepcopy(parent)
        if gap:
            base["pooling"] = None
        morphs = []
        for m in produce_all_morphs(Cnn1DArchitecture(base)):
            a = m.architecture
            if gap and a.get("pooling") is None:
                a["pooling"] = dict(GAP)      # unchanged head; the fork's "add pooling" morph
            morphs.append(a)                  # (GAP -> pooling + Flatten) is kept as it is
        a = deepcopy(parent)                  # switch the head: GAP <-> Flatten
        a["pooling"] = None if gap else dict(GAP)
        morphs.append(a)
        if len(parent["dense_blocks"]) == 1:  # the fork never removes the last hidden layer
            a = deepcopy(parent)
            a["dense_blocks"] = []
            morphs.append(a)
        d = parent.get("head_dropout", 0.0)
        i = DROPOUTS.index(d) if d in DROPOUTS else 0
        for j in (i - 1, i + 1):
            if 0 <= j < len(DROPOUTS):
                a = deepcopy(parent)
                a["head_dropout"] = DROPOUTS[j]
                morphs.append(a)
        return [GapCnn1DArchitecture(a) for a in morphs]

    def to_keras_model(self, arch, input_shape=None, num_classes=None, **kwargs):
        return arch.to_keras_model(input_shape=input_shape or self.input_shape,
                                   num_classes=num_classes or self.num_classes, **kwargs)
