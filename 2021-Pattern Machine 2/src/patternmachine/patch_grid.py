import math
from typing import Tuple

from patternmachine.conv_utils import ConvolutionUtils
from patternmachine.signal_grid import SignalGrid, SignalGridHP
from patternmachine.signal_grid_set import SignalGridSet


class PatchGrid:
    def __init__(self, signal: SignalGridSet, grid_shape: Tuple, coverage_factor=1.0):
        # signal, i.e. composite image, must be a single composite signal, i.e. its grid shape must be (1,)
        assert signal.hp.grid_shape == (1, 1), f"{signal.hp.grid_shape} != (1,1)"
        self.patches: SignalGridSet = self.make_afferent_patches(
            signal=signal, grid_shape=grid_shape, coverage_factor=coverage_factor
        )

    @property
    def patches_info(self):
        return self.patches.patches_info

    @staticmethod
    def make_afferent_patches(
        signal: SignalGridSet, grid_shape: Tuple, coverage_factor: float = 1.0
    ):
        # print("make_afferent_patches")
        # print("  signal", signal.signal_shape)
        # print("  grid_shape", grid_shape)
        # print("  coverage_factor", coverage_factor)

        assert signal.grid_shape == (1, 1)

        if grid_shape == (1, 1):
            # no convolution
            return signal

        for grid_shape_i in grid_shape:
            assert grid_shape_i > 1

        patch_signal_grids = {}

        for name, component_index in signal.components.items():
            # print("  component", name)
            component = signal.components[name]

            patch_shape = tuple(
                [
                    math.ceil(coverage_factor * component.signal_shape[i] / (grid_shape[i]))
                    for i in range(len(grid_shape))
                ]
            )
            # print("    patch_shape", patch_shape)
            stride = tuple(
                [
                    (1.0 * (component.signal_shape[i] - patch_shape[i]) / (grid_shape[i] - 1))
                    for i in range(len(grid_shape))
                ]
            )
            # print("    stride", stride)

            assert component.pixels.shape == component.precision.shape
            patches, patches_info = ConvolutionUtils.conv_slice(
                component.pixels.view(component.hp.signal_shape), patch_shape, stride=stride
            )
            precision_patches, _ = ConvolutionUtils.conv_slice(
                component.precision.view(component.hp.signal_shape), patch_shape, stride=stride
            )
            # print("    patches_info", patches_info)
            sghp = SignalGridHP(grid_shape=patches_info.grid_shape, signal_shape=patch_shape)
            patch_signal_grid = SignalGrid(
                hp=sghp,
                alloc_pixels=False,
                pixels=patches,
                alloc_precision=False,
                precision=precision_patches,
            )
            patch_signal_grid.patches_info = patches_info
            # print("  patch_signal_grid", patch_signal_grid)
            patch_signal_grids[name] = patch_signal_grid

        patches = SignalGridSet.from_signal_grids(patch_signal_grids)
        return patches

    def imshow(self):
        self.patches.imshow()