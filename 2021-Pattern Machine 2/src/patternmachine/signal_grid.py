from typing import Dict, Tuple, Union

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
from config import Config
from patternmachine.conv_utils import ImagePatchCoordinates, ImagePatchesInfo

from patternmachine.signal_utils import SignalUtils
from patternmachine.trace import Trace
from patternmachine.utils import (
    _0213_permute_index,
    gaussian_kernel,
    make_2d_image,
    make_2d_image_alpha,
    precision_to_variance,
    pretty_s,
    show_2d_image_alpha,
    sigmoid_addition,
    inverse,
    sum_gaussian,
)
import patternmachine.utils as utils
from config import Config

import matplotlib.pyplot as plt

from patternmachine.similarity_utils import MAX_PRECISION, MIN_PRECISION, similarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SignalGridHP:
    def __init__(
        self,
        grid_shape: Tuple,
        signal_shape: Tuple,
        init_pixel_scale: float = 0.1,
        init_pixel_value: float = None,
        init_precision_scale: float = 0.1,
        init_precision_value: float = None,
    ):
        self.grid_shape = grid_shape
        self.grid_size = np.prod(self.grid_shape)
        if self.grid_size <= 0:
            raise ValueError("Invalid grid size")

        self.signal_shape = tuple(signal_shape)
        self.signal_size = np.prod(self.signal_shape)
        if self.signal_size <= 0:
            raise ValueError("Invalid signal size")

        self.init_pixel_scale = init_pixel_scale
        self.init_pixel_value = init_pixel_value

        self.init_precision_scale = init_precision_scale
        self.init_precision_value = init_precision_value


class SignalGrid:
    @staticmethod
    def mix2(sg1, sg2):
        result = SignalGrid(hp=sg1.hp, alloc_pixels=False, alloc_precision=False)
        # pdb.set_trace()
        result.pixels = (sg1.pixels * sg1.precision + sg2.pixels * sg2.precision) / (
            sg1.precision + sg2.precision + 0.00001
        )

        scale = sg1.precision * sg2.precision * (sg1.pixels - sg2.pixels).abs()
        result.precision = sigmoid_addition([sg1.precision, sg2.precision]) * inverse(scale)
        return result

    @staticmethod
    def precision_weighted_mix_tensor_lists(stacked_pixels, stacked_precision, signal_grid_hp):
        """
        input: tensors (patch count, tensor count, tensor_size)
        output: (patch count) signal grids with pixels.size == (tensor_size)
        """
        assert signal_grid_hp.grid_size * signal_grid_hp.signal_size == stacked_pixels.shape[-1]
        assert len(stacked_pixels.shape) == 3, "Must be (patch count, tensor count, tensor_size)"

        pixels, precision = sum_gaussian(means=stacked_pixels, precisions=stacked_precision, dim=1)

        result_list = []
        for i in range(stacked_pixels.shape[0]):
            result = SignalGrid(hp=signal_grid_hp, alloc_pixels=False, alloc_precision=False)
            result.pixels = pixels[i].view(signal_grid_hp.grid_size, signal_grid_hp.signal_size)
            result.precision = precision[i].view(
                signal_grid_hp.grid_size, signal_grid_hp.signal_size
            )
            result_list.append(result)

        return result_list

    @staticmethod
    def precision_weighted_mix_tensors(stacked_pixels, stacked_precision, signal_grid_hp):
        pixels, precision = sum_gaussian(
            means=stacked_pixels,
            precisions=stacked_precision,
        )
        # pixels, precision = sum_gaussian(
        #     means=stacked_pixels,
        #     vars=precision_to_variance(stacked_precision),
        #     amplitudes=stacked_precision,
        # )
        result = SignalGrid(hp=signal_grid_hp, alloc_pixels=False, alloc_precision=False)
        result.pixels = pixels.view(signal_grid_hp.grid_size, signal_grid_hp.signal_size)
        result.precision = precision.view(signal_grid_hp.grid_size, signal_grid_hp.signal_size)
        return result

        # # result.pixels = (sg1.pixels * sg1.precision + sg2.pixels * sg2.precision) / (sg1.precision + sg2.precision + 0.00001)
        # sum_precision_weighted_pixels = torch.sum(stacked_pixels * stacked_precision, dim=0)
        # # assert sum_precision_weighted_pixels.shape == signal_shape

        # sum_precision = torch.sum(stacked_precision, dim=0) + 0.00001
        # # assert sum_precision.shape == signal_shape

        # result_pixels = sum_precision_weighted_pixels / sum_precision

        # # scale = sg1.precision * sg2.precision * var([sg1.pixels, sg2.pixels])
        # prod_precision = torch.prod(stacked_precision, dim=0)
        # # assert prod_precision.shape == signal_shape
        # std_pixels = torch.std(
        #     stacked_pixels * 2, unbiased=False, dim=0
        # )  # *2 so that std=1 when pixels and 0 and 1
        # # assert std_pixels.shape == signal_shape
        # scale = prod_precision * std_pixels
        # result_precision = sigmoid_addition(stacked_precision) * inverse(scale)

        # result = SignalGrid(hp=signal_grid_hp, alloc_pixels=False, alloc_precision=False)

        # result.pixels = result_pixels.view(signal_grid_hp.grid_size, signal_grid_hp.signal_size)
        # result.precision = result_precision.view(
        #     signal_grid_hp.grid_size, signal_grid_hp.signal_size
        # )
        # return result

    @staticmethod
    def hallucinate_with_attention(sg_list):
        assert len(sg_list) == 2, "Signal and prediction needed to hallucinate"
        mix = SignalGrid.precision_weighted_mix(sg_list)
        sim = similarity(
            x=sg_list[0].pixels,
            x_precision=sg_list[0].precision,
            y=sg_list[1].pixels,
            y_precision=sg_list[1].precision,
            # precision_based_selectivity=False,
        )

        # attention = inverse(sim / sim.mean())
        attention = inverse(sim)
        attention_weight = 1
        mix.precision = mix.precision * (1 - attention_weight + (attention * attention_weight))
        mix.precision.clamp_(min=MIN_PRECISION, max=MAX_PRECISION)
        return mix

    @staticmethod
    def precision_weighted_mix(sg_list):
        signal_shape = sg_list[0].pixels.shape

        # pdb.set_trace()
        stacked_pixels = torch.stack([sg.pixels for sg in sg_list])
        stacked_precision = torch.stack([sg.precision for sg in sg_list])
        assert stacked_pixels.shape == stacked_precision.shape

        return SignalGrid.precision_weighted_mix_tensors(
            stacked_pixels=stacked_pixels,
            stacked_precision=stacked_precision,
            signal_grid_hp=sg_list[0].hp,
        )

    @staticmethod
    def from_patch(sghp: SignalGridHP, pc: ImagePatchCoordinates, patch: "SignalGrid"):
        sg = SignalGrid(hp=sghp, init_pixel_value=Config.BASE_ACTIVATION, init_precision_value=0.0)
        pixels = sg.pixels[0].view(sg.signal_shape)
        precision = sg.precision[0].view(sg.signal_shape)
        patch_pixels = patch.pixels.view(patch.signal_shape)
        patch_precision = patch.precision.view(patch.signal_shape)

        pixels[
            pc.image_coordinates_from[0] : pc.image_coordinates_to[0] + 1,
            pc.image_coordinates_from[1] : pc.image_coordinates_to[1] + 1,
        ] = patch_pixels[
            pc.patch_coordinates_from[0] : pc.patch_coordinates_to[0] + 1,
            pc.patch_coordinates_from[1] : pc.patch_coordinates_to[1] + 1,
        ]

        precision[
            pc.image_coordinates_from[0] : pc.image_coordinates_to[0] + 1,
            pc.image_coordinates_from[1] : pc.image_coordinates_to[1] + 1,
        ] = patch_precision[
            pc.patch_coordinates_from[0] : pc.patch_coordinates_to[0] + 1,
            pc.patch_coordinates_from[1] : pc.patch_coordinates_to[1] + 1,
        ]

        return sg

    def __init__(
        self,
        hp: SignalGridHP,
        alloc_pixels=True,
        alloc_precision=True,
        pixels=None,
        precision=None,
        init_pixel_value: float = Config.BASE_ACTIVATION,
        init_pixel_kernel: str = utils.KERNEL_RANDOM,  # utils.KERNEL_*
        init_pixel_noise_amplitude: float = 0.0,
        init_precision_value: float = MIN_PRECISION,
        init_precision_kernel: str = utils.KERNEL_RANDOM,  # utils.KERNEL_*
        init_precision_noise_amplitude: float = 0.0,
        patches_info: ImagePatchesInfo = None,
    ):
        self.hp = hp
        self.patches_info: ImagePatchesInfo = patches_info
        if pixels is not None:
            assert pixels.shape == (np.prod(self.hp.grid_shape), np.prod(self.hp.signal_shape))
            self.pixels = Trace(pixels)
        elif alloc_pixels:
            if init_pixel_value is None:
                init_pixel_value = hp.init_pixel_value

            if init_pixel_value is not None:
                self.pixels = (
                    torch.ones((hp.grid_size, hp.signal_size)).to(device) * init_pixel_value
                )
            else:
                self.pixels = utils.make_kernel_grid(
                    kernel_type=init_pixel_kernel,
                    grid_shape=(hp.grid_size, hp.signal_size),
                    signal_shape=hp.signal_shape,
                    scale=hp.init_pixel_scale,
                    noise_amplitude=init_pixel_noise_amplitude,
                )
            self.pixels = Trace(self.pixels)
        else:
            self.pixels = None

        # initialize precision same as pixels, except
        # precision is not Trace()ed
        if precision is not None:
            self.precision = precision
        elif alloc_precision:
            if init_precision_value is None:
                init_precision_value = hp.init_precision_value

            if init_precision_value is not None:
                self.precision = (
                    torch.ones((hp.grid_size, hp.signal_size)).to(device) * init_precision_value
                )
            else:
                self.precision = utils.make_kernel_grid(
                    kernel_type=init_precision_kernel,
                    grid_shape=(hp.grid_size, hp.signal_size),
                    signal_shape=hp.signal_shape,
                    scale=hp.init_precision_scale,
                    noise_amplitude=init_precision_noise_amplitude,
                )

            self.precision = Trace(self.precision)
        else:
            self.precision = None

    @staticmethod
    def temporal_diff(current: "SignalGrid", previous: "SignalGrid"):
        # TODO: Dedup with SignalGridSet.temporal_diff
        component: SignalGrid = current - previous
        # normalize to +1.0 max
        component.pixels /= component.pixels.max() + 0.00001

        # base activation
        component.pixels += Config.BASE_ACTIVATION

        component.pixels.clamp_(min=0.0, max=1.0)

        return component

    def trace_(self, tensor):
        assert self.pixels.shape == tensor.shape
        self.pixels.trace_(tensor)

    @property
    def signal_shape(self):
        return self.hp.signal_shape

    def _as_image(self, pixels: torch.Tensor, grid_shape, signal_shape):
        assert len(grid_shape) == 2 or len(grid_shape) == 4
        assert len(signal_shape) == 2

        if len(grid_shape) == 2:
            pixels = pixels.view(grid_shape + signal_shape)
            pixels = pixels.permute(0, 2, 1, 3)
            pixels = pixels.reshape(
                pixels.shape[0] * pixels.shape[1], pixels.shape[2] * pixels.shape[3]
            )

        elif len(grid_shape) == 4:
            pixels = pixels.view(
                (
                    grid_shape[0],
                    grid_shape[1],
                    grid_shape[2],
                    grid_shape[3],
                    signal_shape[0],
                    signal_shape[1],
                )
            )
            pixels = pixels.permute(0, 2, 4, 1, 3, 5)
            pixels = pixels.reshape(
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2],
                pixels.shape[3] * pixels.shape[4] * pixels.shape[5],
            )

        return pixels

    @property
    def pixels_as_image(self):
        return self._as_image(self.pixels, self.hp.grid_shape, self.signal_shape)

    @property
    def precision_as_image(self):
        return self._as_image(self.precision, self.hp.grid_shape, self.signal_shape)

    @property
    def pixel_contrast(self):
        # How far are the pixels from base activation 0.3?
        px = (self.pixels - Config.BASE_ACTIVATION).abs() / (1 - Config.BASE_ACTIVATION)

        return px

    @property
    def __info__(self):
        return {
            "grid_shape": self.hp.grid_shape,
            "signal_shape": self.hp.signal_shape,
            "pixels": self.pixels,
            "precision": self.precision,
        }

    def imshow(self, title=None):
        pixels = self.pixels_as_image
        precision = self.precision_as_image

        fig_height = pixels.shape[0] / 10
        fig, axs = plt.subplots(
            nrows=1, ncols=3, sharex=True, sharey=True, figsize=(fig_height * 3, fig_height)
        )

        make_2d_image(
            pixels,
            axs[0],
            title="pixels",
        )

        make_2d_image(
            precision,
            axs[1],
            title="precision",
        )

        make_2d_image_alpha(
            pixels,
            axs[2],
            alpha=precision,
            title="composite",
        )

        fig.suptitle(title)
        plt.show()

    def __sub__(self, other: "SignalGrid"):
        # # subtracting means reducing precision if
        # # pixel values are similar
        # result = self.clone()
        # result.precision = self.precision - other.precision * (
        #     1 - (self.pixels - other.pixels).abs()
        # )
        # return result

        # simple pixel subtraction
        # WARNING! May return negative pixel values
        result = self.clone()
        result.pixels = self.pixels - other.pixels
        return result

    def __add__(self, other: "SignalGrid"):
        return SignalGrid.precision_weighted_mix([self, other])

    def __neg__(self):
        return SignalGrid(
            hp=self.hp,
            alloc_pixels=False,
            alloc_precision=False,
            pixels=-self.pixels,
            precision=self.precision.clone(),
            patches_info=self.patches_info,
        )

    def trace_(self, other: Union["SignalGrid", torch.Tensor, Trace]):
        if type(other) in [torch.Tensor, Trace]:
            self.pixels.trace_(other)
        else:
            self.pixels.trace_(other.pixels)
            self.precision.trace_(other.precision)

    def __repr__(self):
        return pretty_s("", self)

    def clone(self):
        return SignalGrid(
            hp=self.hp,
            alloc_pixels=False,
            alloc_precision=False,
            pixels=self.pixels.clone(),
            precision=self.precision.clone(),
            patches_info=self.patches_info,
        )

    def weighted_mixture(
        self, weights: torch.Tensor, patches_info: ImagePatchesInfo, output_sghp: SignalGridHP
    ):
        # for each patch, mix all patterns weighted by activation
        # Example -
        # self.patterns.end.components['one']
        #
        # :SignalGrid
        #   grid_shape = (5, 5, 3, 4)
        #   signal_shape = (2, 3)
        #   pixels:Trace size(300, 6)
        #   precision:Tensor size(300, 6)
        #
        # This represents 5x5 patch grid. Each patch has a bank of 12 (i.e. 3x4) patterns.
        # This component has signal shape of 2x3.
        # For each of the 25 patches,
        #   take 12 patterns of signal shape 2x3
        #   multiply precision of each pattern by activation for that pattern
        #   mix the 12 patterns to get predicted 2x3 signal
        # We will end up with 5x5 grid of 2x3 predicted signals

        assert len(self.hp.grid_shape) == 4, "Implemented only for pattern grids"

        signal_shape = self.hp.signal_shape
        (pattern_count, signal_size) = self.pixels.shape  # (300, 6)
        patch_count = self.hp.grid_shape[0] * self.hp.grid_shape[1]  # 5 * 5 = 25
        patterns_per_patch = self.hp.grid_shape[2] * self.hp.grid_shape[3]  # 3 * 4 = 12
        assert pattern_count == patch_count * patterns_per_patch

        # Reshape pixels and precision
        # (300, 6) -> (25, 12, 6)
        shape = (
            patch_count,
            patterns_per_patch,
            signal_size,
        )
        # pixels = self.pixels.view(shape)  # shape(25, 12, 6)
        # precision = (self.precision * weights.unsqueeze(-1)).view(shape)  # shape(25, 12, 6)

        # pixels = self.pixels.view(
        #     self.hp.grid_shape[0],
        #     self.hp.grid_shape[1],
        #     self.hp.grid_shape[2],
        #     self.hp.grid_shape[3],
        #     signal_size,
        # )  # shape(25, 12, 6)
        # pixels = pixels.permute(0, 2, 1, 3, 4)
        # pixels = pixels.reshape(shape)
        # precision = self.precision * weights.unsqueeze(-1)  # shape(25, 12, 6)
        # precision = precision.view(
        #     self.hp.grid_shape[0],
        #     self.hp.grid_shape[1],
        #     self.hp.grid_shape[2],
        #     self.hp.grid_shape[3],
        #     signal_size,
        # )  # shape(25, 12, 6)
        # precision = precision.permute(0, 2, 1, 3, 4)
        # precision = precision.reshape(shape)
        pixels = self.pixels.view(shape)
        precision = (self.precision * weights.squeeze().unsqueeze(-1)).view(shape)

        signal_grid_hp = SignalGridHP(grid_shape=(1, 1), signal_shape=signal_shape)

        patch_activations = SignalGrid.precision_weighted_mix_tensor_lists(
            stacked_pixels=pixels, stacked_precision=precision, signal_grid_hp=signal_grid_hp
        )
        # patch_activations = [
        #     SignalGrid.precision_weighted_mix_tensors(
        #         stacked_pixels=pixels[patch_index],  # shape(12, 6)
        #         stacked_precision=precision[patch_index],  # shape(12, 6)
        #         signal_grid_hp=signal_grid_hp,
        #     )  # shape(2,3)
        #     for patch_index in range(patch_count)
        # ]  # (25),(2,3) ~= (5, 5, 2, 3)

        # [patch_activations[i].imshow() for i in range(9)]; print()

        # for patch_index in range(patch_count):
        #     print(f"Patch {patch_index} patterns - pixels")
        #     plt.imshow(
        #         self._as_image(
        #             pixels=pixels[patch_index],
        #             grid_shape=self.hp.grid_shape[2:],
        #             signal_shape=self.hp.signal_shape,
        #         )
        #     )
        #     plt.show()
        #     print(f"Patch {patch_index} weights")
        #     plt.imshow(
        #         weights.view(patch_count, patterns_per_patch)[patch_index].view(
        #             self.hp.grid_shape[2:]
        #         )
        #     )
        #     plt.show()

        #     print(f"Patch {patch_index} patterns - precision")
        #     plt.imshow(
        #         self._as_image(
        #             pixels=self.precision.view(shape)[patch_index],
        #             grid_shape=self.hp.grid_shape[2:],
        #             signal_shape=self.hp.signal_shape,
        #         )
        #     )
        #     plt.show()

        #     print(f"Patch {patch_index} patterns - weighted precision")
        #     plt.imshow(
        #         self._as_image(
        #             pixels=precision[patch_index],
        #             grid_shape=self.hp.grid_shape[2:],
        #             signal_shape=self.hp.signal_shape,
        #         )
        #     )
        #     plt.show()

        #     print(f"Patch {patch_index} activation - pixels")
        #     plt.imshow(patch_activations[patch_index].pixels.view(self.hp.signal_shape))
        #     plt.show()

        #     print(f"Patch {patch_index} activation - precision")
        #     plt.imshow(patch_activations[patch_index].precision.view(self.hp.signal_shape))
        #     plt.show()

        # insert each patch at appropriate coordinates into a blank image
        # of the original signal shape

        if patches_info is None:
            # no convolution
            mixed_signal = patch_activations[0]
        else:
            # reverse convolutions back to original signal shape
            assert patches_info.kernel_shape == patch_activations[0].signal_shape
            patches_in_image = []
            for patch_index in range(patch_count):
                patch_coordinates = patches_info.patches[patch_index]
                patch_in_image = SignalGrid.from_patch(
                    sghp=output_sghp,
                    pc=patch_coordinates,
                    patch=patch_activations[patch_index],
                )
                # print(f"Patch {patch_index} in image - pixels")
                # plt.imshow(patch_in_image.pixels.view(output_sghp.signal_shape))
                # plt.show()

                # print(f"Patch {patch_index} in image - precision")
                # plt.imshow(patch_in_image.precision.view(output_sghp.signal_shape))
                # plt.show()

                patches_in_image.append(patch_in_image)

            # [patches_in_image[i].imshow() for i in range(9)]; print()

            # mix all patches that are now copied into blank images
            predicted_signal = SignalGrid.precision_weighted_mix(patches_in_image)
            mixed_signal = predicted_signal
            # print(f"mixed - pixels")
            # plt.imshow(mixed_signal.pixels.view(output_sghp.signal_shape))
            # plt.show()

            # print(f"mixed - precision")
            # plt.imshow(mixed_signal.precision.view(output_sghp.signal_shape))
            # plt.show()

        return mixed_signal