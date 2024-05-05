from typing import Tuple

import numpy as np
import math

from patternmachine.conv_utils import ConvolutionUtils
from patternmachine.signal_grid import SignalGrid, SignalGridHP


class NeighborhoodPatchGrid:
    def __init__(
        self,
        signal: SignalGrid,
        patch_grid_shape: Tuple,
        per_patch_grid_shape: Tuple,
        patch_neighborhood_shape: Tuple,
    ):
        # signal must be a single signal, i.e. its grid shape must be (1,)
        assert signal.hp.grid_shape == (1, 1), f"{signal.hp.grid_shape} != (1,1)"

        patch_shape = tuple(np.multiply(patch_neighborhood_shape, per_patch_grid_shape))

        if patch_shape == signal.signal_shape:
            padding = None
        else:
            padding = tuple([int((i - 1) / 2) for i in patch_shape])
        # padding = tuple(np.multiply(padding_patches, per_patch_grid_shape))
        self.padding = padding
        # print("  padding", padding)

        patches_pixels, patches_info = ConvolutionUtils.conv_slice(
            image=signal.pixels.view(signal.signal_shape),
            kernel_shape=patch_shape,
            patch_grid_shape=patch_grid_shape,
            padding=padding,
        )

        patches_precision = ConvolutionUtils.get_image_patches(  # don't recompute patches_info
            image=signal.precision.view(signal.signal_shape),
            patches_info=patches_info,
        )

        sghp = SignalGridHP(grid_shape=patch_grid_shape, signal_shape=patch_shape)

        self.patches: SignalGrid = SignalGrid(
            hp=sghp,
            alloc_pixels=False,
            pixels=patches_pixels,
            precision=patches_precision,
        )

        self.patches.patches_info = patches_info
