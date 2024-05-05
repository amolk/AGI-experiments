from typing import Tuple

from patternmachine.neighborhood_patch_grid import NeighborhoodPatchGrid
from patternmachine.patch_grid import PatchGrid
from patternmachine.signal_grid import SignalGrid
from patternmachine.signal_grid_set import SignalGridSet


class InputOutputPatchGrid:
    def __init__(
        self,
        patch_grid_shape: Tuple,
        input_csg: SignalGridSet,
        output: SignalGrid,
        output_patch_neighborhood_shape: Tuple,
        per_patch_pattern_grid_shape: Tuple,
        input_coverage_factor=1.0,
    ):
        self.input_patches = PatchGrid(
            signal=input_csg, grid_shape=patch_grid_shape, coverage_factor=input_coverage_factor
        )
        self.output_patches = NeighborhoodPatchGrid(
            signal=output,
            patch_grid_shape=patch_grid_shape,
            per_patch_grid_shape=per_patch_pattern_grid_shape,
            patch_neighborhood_shape=output_patch_neighborhood_shape,
        )

        self.patches = self.input_patches.patches
        self.patches.add_component("__output__", self.output_patches.patches)

    def __repr__(self):
        strs = []
        strs.append("InputOutputPatchGrid")
        strs.append(self.patches.__repr__())
        return "\n".join(strs)