from typing import Tuple

import numpy as np
import torch

from patternmachine.pattern_grid import PatternGrid
from patternmachine.signal_grid import SignalGrid
from patternmachine.signal_grid_set import SignalGridSet


class LocalLearning:
    def __init__(
        self,
        input_csg: SignalGridSet,
        output: SignalGrid,
        patch_grid_shape: Tuple,
        per_patch_pattern_grid_shape: Tuple,
        patterns: PatternGrid,
    ):
        # Output shape example (1, 2x3x4x5), where patch_grid_shape is (2x4) and per_patch_pattern_grid_shape is (3x5)
        assert len(patch_grid_shape) == len(per_patch_pattern_grid_shape)
        assert output.pixels.shape[0] == 1
        assert output.pixels.shape[2] == np.dot(patch_grid_shape, per_patch_pattern_grid_shape)

        activation_shape = torch.stack(
            (torch.tensor(patch_grid_shape), torch.tensor(per_patch_pattern_grid_shape)), dim=1
        ).view(
            -1
        )  # e.g. (2, 3, 4, 5)
        activation = output.pixels.view(activation_shape)
