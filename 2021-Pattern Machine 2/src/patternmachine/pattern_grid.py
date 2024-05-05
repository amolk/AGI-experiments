from typing import Dict, List, Tuple

from IPython.display import display
import numpy as np
import torch
from patternmachine.conv_utils import ImagePatchesInfo

from patternmachine.signal_grid_set import SignalGridHP, SignalGridSet, SignalGridSetHP
from patternmachine.utils import pretty_s, show_1d_image
import patternmachine.utils as utils

print = display

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PatternGridHP:
    def __init__(
        self,
        grid_shape: Tuple,
        pattern_signal_set_shape: Dict,
        init_pixel_values: Dict = None,
        init_pixel_kernels: Dict = None,
        init_pixel_noise_amplitude: Dict = None,
        init_precision_values: Dict = None,
        init_precision_kernels: Dict = None,
        init_precision_noise_amplitude: Dict = None,
    ) -> None:

        self.grid_shape = grid_shape
        self.grid_size = np.prod(self.grid_shape)

        self.signal_grid_set_hp = SignalGridSetHP(
            hps={
                name: SignalGridHP(grid_shape=grid_shape, signal_shape=signal_shape)
                for name, signal_shape in pattern_signal_set_shape.items()
            }
        )

        self.init_pixel_values = init_pixel_values
        self.init_precision_values = init_precision_values

        if init_pixel_kernels is None:
            init_pixel_kernels = {
                name: utils.KERNEL_RANDOM for name in pattern_signal_set_shape.keys()
            }
        self.init_pixel_kernels = init_pixel_kernels

        if init_pixel_noise_amplitude is None:
            init_pixel_noise_amplitude = {name: 0.0 for name in pattern_signal_set_shape.keys()}
        self.init_pixel_noise_amplitude = init_pixel_noise_amplitude

        if init_precision_kernels is None:
            init_precision_kernels = {
                name: utils.KERNEL_RANDOM for name in pattern_signal_set_shape.keys()
            }
        self.init_precision_kernels = init_precision_kernels

        if init_precision_noise_amplitude is None:
            init_precision_noise_amplitude = {
                name: 0.0 for name in pattern_signal_set_shape.keys()
            }
        self.init_precision_noise_amplitude = init_precision_noise_amplitude


class PatternGrid:
    def __init__(self, hp: PatternGridHP):
        self.hp = hp

        self.begin: SignalGridSet = SignalGridSet(
            hp.signal_grid_set_hp,
            init_pixel_values=hp.init_pixel_values,
            init_pixel_kernels=hp.init_pixel_kernels,
            init_pixel_noise_amplitude=hp.init_pixel_noise_amplitude,
            init_precision_values=hp.init_precision_values,
            init_precision_kernels=hp.init_precision_kernels,
            init_precision_noise_amplitude=hp.init_precision_noise_amplitude,
        )  # Trajectory begin

        self.end: SignalGridSet = SignalGridSet(
            hp.signal_grid_set_hp,
            init_pixel_values=hp.init_pixel_values,
            init_pixel_kernels=hp.init_pixel_kernels,
            init_pixel_noise_amplitude=hp.init_pixel_noise_amplitude,
            init_precision_values=hp.init_precision_values,
            init_precision_kernels=hp.init_precision_kernels,
            init_precision_noise_amplitude=hp.init_precision_noise_amplitude,
        )  # Trajectory end

        # noinspection PyTypeChecker
        self.alpha = torch.ones((hp.grid_size,)).to(device)

        self.init_pattern_to_patch_map()

    # each pattern uses a certain afferent input patch
    # find a mapping between pattern index and patch index
    def init_pattern_to_patch_map(self):
        patch_grid_shape = self.hp.grid_shape[0:2]
        per_patch_pattern_grid_shape = self.hp.grid_shape[2:4]
        pattern_to_patch_map = torch.tensor(
            range(patch_grid_shape[0] * patch_grid_shape[1])
        ).unsqueeze(dim=-1)
        pattern_to_patch_map = pattern_to_patch_map.expand(
            (-1, per_patch_pattern_grid_shape[0] * per_patch_pattern_grid_shape[1])
        )
        pattern_to_patch_map = pattern_to_patch_map.view(self.hp.grid_shape)
        pattern_to_patch_map = pattern_to_patch_map.permute(0, 2, 1, 3).reshape(
            patch_grid_shape[0] * per_patch_pattern_grid_shape[0],
            patch_grid_shape[1] * per_patch_pattern_grid_shape[1],
        )
        pattern_to_patch_map = pattern_to_patch_map.view(-1)
        self.pattern_to_patch_map = pattern_to_patch_map

    def __repr__(self):
        return pretty_s("", self)

    @property
    def pixels(self):
        return {
            "pixels_begin": self.begin.pixels,
            "precision_begin": self.begin.precision,
            "pixels_end": self.end.pixels,
            "precision_end": self.end.precision,
        }

    @property
    def pixels_as_image(self):
        return {
            "pixels_begin": self.begin.pixels_as_image,
            "precision_begin": self.begin.precision_as_image,
            "pixels_end": self.end.pixels_as_image,
            "precision_end": self.end.precision_as_image,
        }

    def mix_patterns_by_activation(
        self,
        activation: torch.Tensor,
        patches_info: Dict[str, ImagePatchesInfo],
        output_sgshp: SignalGridSetHP,
    ):
        return self.end.weighted_mixture(
            weights=activation, patches_info=patches_info, output_sgshp=output_sgshp
        )

    def imshow(self):
        self.begin.imshow(title="begin")
        self.end.imshow(title="end")

    # def interpolate(self, pattern_index: int, fraction: float):
    #     assert fraction >= 0.0
    #     assert fraction <= 1.0

    #     pixels1 = self.pixels_begin[pattern_index].pixels
    #     precision1 = self.precision_begin[pattern_index].pixels
    #     pixels2 = self.pixels_end[pattern_index].pixels
    #     precision2 = self.precision_end[pattern_index].pixels

    #     if fraction < 0.01:
    #         return [pixels1, precision1]

    #     if fraction > 0.99:
    #         return [pixels2, precision2]

    #     pixels = {}
    #     precision = {}
    #     for name in pixels1:
    #         pix, pre = self.ttt_interpolate(
    #             [pixels1[name], precision1[name]], [pixels2[name], precision2[name]], fraction
    #         )
    #         pixels[name] = pix
    #         precision[name] = pre

    #     return [pixels, precision]

    # def ttt_interpolate(self, t1: List[torch.tensor], t2: List[torch.tensor], fraction: float):
    #     assert t1[0].shape == t1[1].shape
    #     assert t2[0].shape == t2[1].shape
    #     assert t1[0].shape == t2[1].shape

    #     mask1 = t1[0] > fraction + 0.2
    #     mask2 = (t2[0] < fraction - 0.2) & (t2[0] > 0.01)
    #     t_pixels = torch.zeros_like(t1[0])

    #     masked_t2 = t2[0][mask2]
    #     t_pixels[mask2] += masked_t2 / (1 - fraction)
    #     t_pixels /= torch.max(t_pixels) + 0.01

    #     # print("fraction", fraction)
    #     # print("t1[0]")
    #     # show_1d_image(t1[0].squeeze())
    #     # print("mask1")
    #     # show_1d_image(mask1.squeeze())
    #     t_pixels += t1[0] * (1 - fraction)
    #     # print("t_pixels")
    #     # show_1d_image(t_pixels.squeeze())
    #     # print("t1_pixels", t1_pixels)

    #     t_pixels.clamp(0.0, 1.0)
    #     # print("t2_pixels", t2_pixels)

    #     return [t_pixels, None]
