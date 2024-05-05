from collections import deque
from copy import deepcopy
import math
import pdb
from typing import Dict, List, Tuple
import random

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
import torchvision.transforms as T
from config import Config

from patternmachine.clock import Clock
from patternmachine.conv_utils import ConvolutionUtils
from patternmachine.input_output_patch_grid import InputOutputPatchGrid
from patternmachine.patch_grid import PatchGrid
from patternmachine.pattern_grid import PatternGrid, PatternGridHP
from patternmachine.pattern_similarity import PatternSimilarity
from patternmachine.signal_grid import SignalGrid, SignalGridHP
from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.trace import Trace
from patternmachine.utils import (
    _0213_permute_index,
    bounded_contrast_enhancement,
    gaussian_kernel,
    inverse,
    make_2d_image,
    sum_gaussian,
    variance_to_precision,
    precision_to_variance,
    normalize,
)
import patternmachine.utils as utils
import matplotlib.pyplot as plt

from patternmachine.similarity_utils import MAX_PRECISION, MIN_PRECISION

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Strategies to find winners
FIND_WINNERS_SET_ALL = 0
FIND_WINNERS_SET_ARBITRARY = 1
FIND_WINNERS_SET_PER_PATCH = 2

FIND_WINNERS_COUNT_EXACT_N = 0
FIND_WINNERS_COUNT_UPTO_N = 1

FIND_WINNERS_BY_MAX_ACTIVATION = 0
FIND_WINNERS_BY_MULTINOMIAL = 1


class LayerHP:
    def __init__(
        self,
        input_signal_shapes: Dict[str, Tuple],
        input_coverage_factor: float,
        patch_grid_shape: Tuple,
        per_patch_pattern_grid_shape: Tuple,
        output_patch_neighborhood_shape: Tuple,
        output_decay=0.5,
        pattern_init_pixel_values: Dict = None,
        pattern_init_pixel_noise_amplitude: Dict = None,
        pattern_init_precision_values: Dict = None,
        pattern_init_precision_noise_amplitude: Dict = None,
        alpha: float = 0.01,
        # tau: int = 2,
    ):
        assert len(patch_grid_shape) == len(
            per_patch_pattern_grid_shape
        ), "This is so that output can be flattened"
        assert len(patch_grid_shape) == len(
            output_patch_neighborhood_shape
        ), "Output patch neighborhood is a subset of patch grid, so number of dimensions must match"
        # assert len(patch_grid_shape) == 2, "Currently support for 2D"

        self.alpha = alpha
        self.enable_gaussian_precision = False
        self.enhance_activation_contrast = True
        self.enhance_activation_contrast_target = 0.9
        self.enhance_activation_contrast_power = 1000
        self.neighborhood_smoothing = 1.0
        self.low_precision_near_base_activation = True
        self.temporal_neighborhood_activation_handicap_weight = 1.0
        self.temporal_neighborhood_activation_handicap_additive = False
        self.input_signal_shapes: Dict[str, Tuple] = input_signal_shapes
        self.input_coverage_factor: float = input_coverage_factor
        self.patch_grid_shape: Tuple = patch_grid_shape
        self.per_patch_pattern_grid_shape: Tuple = per_patch_pattern_grid_shape
        self.output_patch_neighborhood_shape = output_patch_neighborhood_shape
        self.output_neighborhood_shape: Tuple = np.multiply(
            output_patch_neighborhood_shape, per_patch_pattern_grid_shape
        )
        self.output_decay: float = output_decay
        # self.tau = tau

        for size in self.output_patch_neighborhood_shape:
            assert (
                size % 2 == 1
            ), "Output patch neighborhood shape must be odd, so can be centered around specific output activation"

        # Get input patch shapes
        sample_input_signal = {
            name: torch.ones(shape) for name, shape in input_signal_shapes.items()
        }
        sample_input: SignalGridSet = SignalGridSet.from_pixels_list(sample_input_signal)
        # print("sample_input", sample_input)
        # print("patch_grid_shape", patch_grid_shape)
        # print("input_coverage_factor", input_coverage_factor)
        patches = PatchGrid.make_afferent_patches(
            signal=sample_input, grid_shape=patch_grid_shape, coverage_factor=input_coverage_factor
        )

        # Patterns -
        # each patch in the patch grid will get per_patch_pattern_grid_shape patterns, so patterns will be of shape -
        # (patch_grid_shape,) + (per_patch_pattern_grid_shape shape,) + (pattern_signal_set_shape,)
        pattern_signal_set_shape = patches.signal_shape.copy()

        # input signals + output signal
        pattern_signal_set_shape["__output__"] = self.output_neighborhood_shape

        pattern_grid_shape = patch_grid_shape + per_patch_pattern_grid_shape  # 2D, 2D = 4D

        pattern_init_pixel_kernels = {
            component_name: utils.KERNEL_GAUSSIAN for component_name in input_signal_shapes
        }
        pattern_init_pixel_kernels["__output__"] = utils.KERNEL_DOG

        pattern_init_precision_kernels = {
            component_name: utils.KERNEL_DOG for component_name in input_signal_shapes
        }
        pattern_init_precision_kernels["__output__"] = utils.KERNEL_DOG

        self.pattern_grid_hp: PatternGridHP = PatternGridHP(
            grid_shape=pattern_grid_shape,
            pattern_signal_set_shape=pattern_signal_set_shape,
            init_pixel_values=pattern_init_pixel_values,
            init_pixel_kernels=pattern_init_pixel_kernels,
            init_pixel_noise_amplitude=pattern_init_pixel_noise_amplitude,
            init_precision_values=pattern_init_precision_values,
            init_precision_kernels=pattern_init_precision_kernels,
            init_precision_noise_amplitude=pattern_init_precision_noise_amplitude,
        )

        # Output is flattened 2D version of the 4D pattern_grid_shape
        output_shape = np.multiply(patch_grid_shape, per_patch_pattern_grid_shape)

        self.output_hp: SignalGridHP = SignalGridHP(
            grid_shape=(1, 1), signal_shape=output_shape, init_pixel_scale=0.0
        )

        # print("Output shape", output_shape)

        # # input grid HP
        # hps = [
        #   SignalGridHP(
        #       grid_shape=output_shape, # Each grid element in input grid produces 1 pixel of output
        #       signal_shape=input_component.signal_shape)
        #   for input_component in patches.components # each component of input
        # ]
        # self.input_grid_hp = SignalGridSetHP(hps=hps.copy())

        # # pattern HP
        # hps.append(SignalGridHP(
        #     grid_shape=output_shape,
        #     signal_shape=output_neighborhood_shape))

        # self.pattern_grid_hp = PatternGridHP(
        #     grid_shape=pattern_grid_shape,
        #     composite_signal_grid_hp=SignalGridSetHP(hps=[
        #       SignalGridHP(grid_shape=pattern_grid_shape, signal_shape=component.signal_shape)
        #      for component in hps]))


class Layer:
    def __init__(self, hp: LayerHP, inputs: Dict = None):
        self.hp = hp
        self.patterns = PatternGrid(hp=self.hp.pattern_grid_hp)
        # self.pattern_win_counts = {}
        self.output = SignalGrid(hp.output_hp, init_pixel_value=0, init_precision_value=1)
        self.output.pixels.epsilon = self.hp.output_decay
        self.raw_output = SignalGrid(hp.output_hp, init_pixel_value=0, init_precision_value=1)
        self.raw_output.pixels.epsilon = self.hp.output_decay
        self.td_output = SignalGrid(hp.output_hp, init_pixel_value=0, init_precision_value=1)
        self.activation = torch.zeros(self.output.pixels.shape[1])
        self.activation_i = torch.zeros(self.output.pixels.shape[1])
        self.activation_i_trace = Trace(
            self.activation_i.view(self.output.signal_shape), epsilon=0.3
        )
        self.debug = False
        # self.bottom_up_similarity_end = None
        # self.input_output_patch_grid_buffer = deque(maxlen=self.hp.tau)
        self.inputs = inputs
        self.top_down_winner_index = None
        self.previous_trace_input = None
        self.td_input = None
        self.mu = None
        self.mu_bar = None

        if self.output.pixels.nelement() > 1:
            # handicap_hp = deepcopy(hp.output_hp)
            # handicap_hp.init_pixel_value = 1.0
            # self.temporal_neighborhood_activation_handicap = SignalGrid(handicap_hp)
            # self.temporal_neighborhood_activation_handicap.pixels.epsilon = 0.0
            self.temporal_neighborhood_activation_handicap = Trace(
                torch.zeros(hp.output_hp.signal_shape)
            )
        self.previous_output = None
        self.blurrer = T.GaussianBlur(
            kernel_size=[i * 2 - 1 for i in self.hp.per_patch_pattern_grid_shape],
            sigma=1.5,
        )

        self.previous_input_output_patch_grid = None
        self.previous_raw_input_output_patch_grid = None

    def next(self, clock: Clock):
        # process bottom up input only on first clock tick in the cycle
        if clock.t % (clock.tau + 1) == 0:
            input = SignalGridSet.from_signal_grids(
                {k: v.current(clock) for k, v in self.inputs.items()}
            )
            self.forward(input)

        return self.current(clock)

    def current(self, clock: Clock):
        return self.patterns.begin[self.top_down_winner_index]

    def initialize_trace_input(self, trace_input: SignalGridSet) -> None:
        self.previous_trace_input = trace_input
        self.td_input = trace_input.clone()
        self.td_input.components["mu"].pixels *= 0
        self.mu = trace_input.clone()
        for component in self.mu.components.values():
            component.pixels = component.pixels * 0.0

    def forward(self, trace_input: SignalGridSet, learn: True):
        assert trace_input.grid_shape == (1, 1)  # single signal input

        # Layer needs temporal difference signal, so skip first frame
        if self.previous_trace_input is None:
            self.initialize_trace_input(trace_input)
            return

        # td_input = TD(trace_input, previous_trace_input)
        td_input: SignalGridSet = SignalGridSet.temporal_diff(
            current=trace_input, previous=self.previous_trace_input
        )

        ## if use trace of td_input
        # self.td_input.trace_(td_input)
        ## else
        self.td_input = td_input
        self.previous_trace_input = trace_input.clone()

        # mix input with previous prediction
        if self.mu_bar is None:
            self.mu = self.td_input
        else:
            self.mu = SignalGridSet.hallucinate_with_attention([self.td_input, self.mu_bar])
            # self.mu = self.td_input

        current_input_output_patch_grid = InputOutputPatchGrid(
            patch_grid_shape=self.hp.patch_grid_shape,
            input_csg=self.mu,
            output=self.td_output,
            output_patch_neighborhood_shape=self.hp.output_patch_neighborhood_shape,
            per_patch_pattern_grid_shape=self.hp.per_patch_pattern_grid_shape,
            input_coverage_factor=self.hp.input_coverage_factor,
        )

        current_raw_input_output_patch_grid = InputOutputPatchGrid(
            patch_grid_shape=self.hp.patch_grid_shape,
            input_csg=self.td_input,
            output=self.td_output,
            output_patch_neighborhood_shape=self.hp.output_patch_neighborhood_shape,
            per_patch_pattern_grid_shape=self.hp.per_patch_pattern_grid_shape,
            input_coverage_factor=self.hp.input_coverage_factor,
        )

        if learn:
            # Learn
            self.learn(
                previous_input_output_patch_grid=self.previous_input_output_patch_grid,
                previous_raw_input_output_patch_grid=self.previous_raw_input_output_patch_grid,
                current_input_output_patch_grid=current_input_output_patch_grid,
                current_raw_input_output_patch_grid=current_raw_input_output_patch_grid,
            )

        # Update top-down prediction
        self.predict(current_input_output_patch_grid)

        self.previous_input_output_patch_grid = current_input_output_patch_grid
        self.previous_raw_input_output_patch_grid = current_raw_input_output_patch_grid

    def learn(
        self,
        previous_input_output_patch_grid,
        previous_raw_input_output_patch_grid,
        current_input_output_patch_grid,
        current_raw_input_output_patch_grid,
    ):
        if previous_input_output_patch_grid is None:
            return

        # Find winning pattern such that
        #    previous_input_output_patch_grid == patterns.begin
        #    current_input_output_patch_grid  == patterns.end
        # and update that pattern

        # compute similarity
        pattern_similarity_begin = PatternSimilarity(
            signal=previous_input_output_patch_grid.patches, patterns=self.patterns.begin
        )
        pattern_similarity_end = PatternSimilarity(
            signal=current_input_output_patch_grid.patches, patterns=self.patterns.end
        )

        # activation in image indices
        _, activation, _, raw_activation = self.compute_activation(
            (pattern_similarity_begin.sim * pattern_similarity_end.sim).pow(0.5),
            self.previous_temporal_neighborhood_activation_handicap,
        )

        # # find winners
        # winner_indices_i = self.find_winners(activation_i, strategy_set=FIND_WINNERS_SET_PER_PATCH)

        # self.learn_update_patterns(
        #     winner_indices_i=winner_indices_i,
        #     input_output_patch_grid_begin=previous_input_output_patch_grid,
        #     input_output_patch_grid_end=current_raw_input_output_patch_grid,
        # )

        self.learn_update_patterns_2(
            activation=activation,
            raw_activation=raw_activation,
            input_output_patch_grid_begin=previous_raw_input_output_patch_grid,
            input_output_patch_grid_end=current_raw_input_output_patch_grid,
        )

    def predict(self, current_input_output_patch_grid):
        # compute similarity
        pattern_similarity = PatternSimilarity(
            signal=current_input_output_patch_grid.patches,
            patterns=self.patterns.begin,
        )

        # activation in image indices
        activation_i, activation, raw_activation_i, raw_activation = self.compute_activation(
            pattern_similarity.sim, self.temporal_neighborhood_activation_handicap
        )

        # update output
        activation_i = activation_i.view(-1)
        raw_activation_i = raw_activation_i.view(-1)
        previous_output = self.output.clone()
        self.output.trace_(activation_i)
        self.raw_output.trace_(raw_activation_i)
        self.td_output: SignalGrid = SignalGrid.temporal_diff(
            current=self.output, previous=previous_output
        )

        # update handicap that makes a sting of neighboring patterns
        # get acitivate in sequence
        self.update_temporal_neighborhood_activation_handicap()

        # next input prediction
        self.new_mu_bar = self.patterns.mix_patterns_by_activation(
            activation=activation.view(-1),
            patches_info=current_input_output_patch_grid.input_patches.patches_info,
            output_sgshp=self.mu.hp,
        )
        if self.mu_bar is None:
            self.mu_bar = self.new_mu_bar
        else:
            self.mu_bar.trace_(self.new_mu_bar)

    def compute_activation(self, sim, temporal_neighborhood_activation_handicap) -> torch.Tensor:
        assert (
            sim.nelement()
            == temporal_neighborhood_activation_handicap.nelement()
            == self.output.pixels.nelement()
        )

        sim_i = self.activation_signal_to_image(sim).view(self.output.signal_shape)

        # neighborhood influence
        if self.hp.temporal_neighborhood_activation_handicap_additive:
            activation_i = (
                sim_i
                + temporal_neighborhood_activation_handicap
                * self.hp.temporal_neighborhood_activation_handicap_weight
            )
        else:
            activation_i = sim_i * (
                temporal_neighborhood_activation_handicap.clamp(min=0, max=1)
                * self.hp.temporal_neighborhood_activation_handicap_weight
                + (1 - self.hp.temporal_neighborhood_activation_handicap_weight)
            )

        # refractory / rate limiting
        # print("output")
        # print(output)

        # return bounded_exponential(activation.clamp_(min=0, max=1))
        # activation = bounded_contrast_enhancement(activation)
        activation_i.clamp_(min=0, max=1)

        activation = self.activation_image_to_signal(activation_i)

        if self.hp.enhance_activation_contrast:
            contrast_enhanced_activation = activation + 0.000001
            single_pattern_per_patch_mode = contrast_enhanced_activation.shape[-1] == 1
            if single_pattern_per_patch_mode:
                assert (
                    activation.shape[0]
                    == self.hp.patch_grid_shape[0] * self.hp.patch_grid_shape[1]
                )

                contrast_enhanced_activation = contrast_enhanced_activation.view(
                    self.hp.patch_grid_shape
                )

            iter = 0
            while True:
                contrast_enhanced_activation = bounded_contrast_enhancement(
                    contrast_enhanced_activation,
                    dim=1,
                    power=self.hp.enhance_activation_contrast_power,
                )
                # if iter == 0:
                #     activation = contrast_enhanced_activation
                top2 = contrast_enhanced_activation.topk(2, dim=-1).values[0]
                if top2[0] == 0.0:
                    contrast_enhanced_activation *= 0
                    contrast_enhanced_activation[
                        0, random.randint(0, contrast_enhanced_activation.shape[1] - 1)
                    ] = 1.0
                    break
                contrast = (top2[0] - top2[1]).abs() / top2.sum()
                if contrast.min() > self.hp.enhance_activation_contrast_target:
                    break
                iter += 1
                if iter > 10:
                    break

            if single_pattern_per_patch_mode:
                contrast_enhanced_activation = contrast_enhanced_activation.view(activation.shape)

            contrast_enhanced_activation_i = self.activation_signal_to_image(
                contrast_enhanced_activation
            )

            activation_i = self.activation_signal_to_image(activation)

            return (
                contrast_enhanced_activation_i,
                contrast_enhanced_activation,
                activation_i,
                activation,
            )
        else:
            return (
                activation_i,
                activation,
                activation_i,
                activation,
            )

    def activation_signal_to_image(self, act):
        return (
            act.view(self.hp.patch_grid_shape + self.hp.per_patch_pattern_grid_shape)
            .permute(0, 2, 1, 3)
            .reshape(
                self.hp.patch_grid_shape[0] * self.hp.per_patch_pattern_grid_shape[0],
                self.hp.patch_grid_shape[1] * self.hp.per_patch_pattern_grid_shape[1],
            )
        )

    def activation_image_to_signal(self, act):
        shape = (
            self.hp.patch_grid_shape[0],
            self.hp.per_patch_pattern_grid_shape[0],
            self.hp.patch_grid_shape[1],
            self.hp.per_patch_pattern_grid_shape[1],
        )
        return (
            act.view(shape)
            .permute(0, 2, 1, 3)
            .reshape(
                self.hp.patch_grid_shape[0] * self.hp.patch_grid_shape[1],
                self.hp.per_patch_pattern_grid_shape[0] * self.hp.per_patch_pattern_grid_shape[1],
            )
        )

    def activation_index_image_to_signal(self, i):
        a = self.hp.patch_grid_shape[0]
        b = self.hp.per_patch_pattern_grid_shape[0]
        c = self.hp.patch_grid_shape[1]
        d = self.hp.per_patch_pattern_grid_shape[1]

        return _0213_permute_index(i, a, b, c, d)

    def activation_index_signal_to_image(self, i):
        a = self.hp.patch_grid_shape[0]
        b = self.hp.patch_grid_shape[1]
        c = self.hp.per_patch_pattern_grid_shape[0]
        d = self.hp.per_patch_pattern_grid_shape[1]

        return _0213_permute_index(i, a, b, c, d)

    def learn_update_patterns_2(
        self,
        activation: torch.Tensor,
        raw_activation: torch.Tensor,
        input_output_patch_grid_begin: InputOutputPatchGrid,
        input_output_patch_grid_end: InputOutputPatchGrid,
    ) -> None:
        for component_name in self.patterns.begin.components:
            self._update_patterns_2(
                component_name=component_name,
                input_output_patch_grid=input_output_patch_grid_begin,
                activation=activation,
                raw_activation=raw_activation,
                patterns=self.patterns.begin,
            )

            self._update_patterns_2(
                component_name=component_name,
                input_output_patch_grid=input_output_patch_grid_end,
                activation=activation,
                raw_activation=raw_activation,
                patterns=self.patterns.end,
            )

    def _update_patterns_2(
        self,
        component_name,
        input_output_patch_grid: InputOutputPatchGrid,
        activation,
        raw_activation,
        patterns,
    ):
        per_patch_pattern_count = (
            self.hp.per_patch_pattern_grid_shape[0] * self.hp.per_patch_pattern_grid_shape[1]
        )

        component = patterns.components[component_name]
        patch_component = input_output_patch_grid.input_patches.patches.components[component_name]

        shape = (
            component.hp.grid_shape[0] * component.hp.grid_shape[1],
            component.hp.grid_shape[2] * component.hp.grid_shape[3],
            component.hp.signal_shape[0] * component.hp.signal_shape[1],
        )

        # precision weighted mix patch into pattern
        stacked_pixels = torch.stack(
            [
                component.pixels.view(shape),
                patch_component.pixels.unsqueeze(dim=1).expand(shape),
            ]
        )

        stacked_precision = torch.stack(
            [
                component.precision.view(shape),
                patch_component.precision.unsqueeze(dim=1).expand(shape),
            ]
        )

        pixels, _ = sum_gaussian(
            means=stacked_pixels, precisions=stacked_precision, compuate_precision=False
        )
        precision = (
            1 - (pixels - stacked_pixels[0]).abs()
        )  # inverse((pixels - stacked_pixels[0]).abs())

        pixels = pixels.view(-1, shape[-1])
        precision = precision.view(-1, shape[-1])

        # don't increase precision for pixels close to base activation level (i.e. noise level)
        if self.hp.low_precision_near_base_activation:
            dist_from_base_activation = (patch_component.pixels - Config.BASE_ACTIVATION).abs() / (
                1 - Config.BASE_ACTIVATION
            )
            precision = precision * dist_from_base_activation.unsqueeze(dim=1).expand(
                shape
            ).reshape(-1, shape[-1])

        # # More precision in local neighborhood
        # if self.hp.enable_gaussian_precision:
        #     gc = gaussian_kernel(
        #         l=component.signal_shape[0], sig=component.signal_shape[0] * 0.3
        #     ).view(-1)
        #     precision = precision * gc

        activation = activation.view(shape[0:2])
        activation = normalize(activation, dim=-1)
        activation_i = self.activation_signal_to_image(activation)
        if self.hp.neighborhood_smoothing is not None:
            blurred_activation_i = (
                self.blurrer(activation_i.unsqueeze(dim=0)).squeeze(dim=0) + activation_i * 0.5
            ).clamp_max(1.0)
            activation_i = (
                activation_i * (1 - self.hp.neighborhood_smoothing)
                + blurred_activation_i * self.hp.neighborhood_smoothing
            )
        activation = self.activation_image_to_signal(activation_i)
        alpha = activation.unsqueeze(-1) * self.hp.alpha * normalize(raw_activation).unsqueeze(-1)
        alpha = alpha.view(-1, 1)

        component.pixels = component.pixels * (1 - alpha) + alpha * pixels

        component.precision = (component.precision * (1 - alpha) + alpha * precision).clamp_(
            min=MIN_PRECISION, max=MAX_PRECISION
        )

    def update_temporal_neighborhood_activation_handicap(self):
        if self.output.pixels.nelement() == 0:
            return

        self.previous_temporal_neighborhood_activation_handicap = (
            self.temporal_neighborhood_activation_handicap + 0.000001  # clone
        )

        # output = (
        #     normalize(self.output.pixels.clone()).view(self.output.signal_shape).unsqueeze(dim=0)
        # )
        output = self.output.pixels.view(self.output.signal_shape).unsqueeze(dim=0)

        if self.previous_output is not None:
            blurred_output = self.blurrer(output) + output
            blurred_previous_output = self.blurrer(self.previous_output) + self.previous_output

            self.temporal_neighborhood_activation_handicap.trace_(
                self.blurrer(output) - output - self.blurrer(self.previous_output)
            )

            # fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(25, 5))
            # make_2d_image(output[0] + 0.001, axs[0, 0], title="output")
            # make_2d_image(
            #     self.blurrer(self.previous_output)[0] + 0.001,
            #     axs[0, 1],
            #     title="blurred previous output",
            # )
            # make_2d_image(self.blurrer(output)[0] + 0.001, axs[0, 2], title="blurred output")
            # make_2d_image(
            #     (self.blurrer(output) - output)[0] + 0.001,
            #     axs[1, 0],
            #     title="blurred output - output",
            # )
            # make_2d_image(
            #     self.temporal_neighborhood_activation_handicap[0] + 0.001,
            #     axs[1, 1],
            #     title="handicap",
            # )
            # plt.show()

        else:
            self.temporal_neighborhood_activation_handicap.trace_(self.blurrer(output) - output)

        self.temporal_neighborhood_activation_handicap = (
            self.temporal_neighborhood_activation_handicap.view(self.output.signal_shape)
        )
        self.previous_output = output

    def find_winners(
        self,
        activation: torch.Tensor,
        max_winners: int = None,
        strategy_set=FIND_WINNERS_SET_ARBITRARY,
        strategy_count=FIND_WINNERS_COUNT_UPTO_N,
        strategy_by=FIND_WINNERS_BY_MAX_ACTIVATION,
    ):
        assert activation.shape == self.output.signal_shape

        if max_winners is None:
            max_winners = torch.prod(
                torch.tensor(self.hp.patch_grid_shape)
            ).item()  # 1 winner per patch

        # collect non-neighboring winner indices
        winner_indices = []

        if strategy_set == FIND_WINNERS_SET_ALL:
            # ignore strategy_count and strategy_by because everyone's a winner
            winner_indices = torch.range(0, self.activation.nelement() - 1).long()
        elif strategy_set == FIND_WINNERS_SET_ARBITRARY:
            act = (activation * 1.0).view(-1)  # make copy of tensor
            while len(winner_indices) < max_winners:
                if strategy_by == FIND_WINNERS_BY_MAX_ACTIVATION:
                    v, i = act.max(dim=0)
                elif strategy_by == FIND_WINNERS_BY_MULTINOMIAL:
                    i = torch.multinomial(act, 1)[0].item()
                    v = act[i]
                else:
                    raise ValueError(f"Find winners strategy_by {strategy_by} is invalid")

                if strategy_count == FIND_WINNERS_COUNT_EXACT_N:
                    pass
                elif strategy_count == FIND_WINNERS_COUNT_UPTO_N:
                    if v < 0.0001:
                        if len(winner_indices) == 0:
                            winner_indices.append(i.item())
                        break
                else:
                    raise ValueError(f"Find winners strategy_count {strategy_count} is invalid")

                winner_indices.append(i.item())
                o = torch.zeros_like(act)
                o[i] = v
                o = self.blurrer(o.view(self.output.signal_shape).unsqueeze(dim=0)).squeeze(dim=0)
                o = o.view(-1)
                o = o * v / o[i]
                act = (act - o).clamp(min=0, max=1)
        elif strategy_set == FIND_WINNERS_SET_PER_PATCH:
            # convert activation to signal space
            act = self.activation_image_to_signal(activation)

            act = act.view(
                -1,
                self.hp.per_patch_pattern_grid_shape[0] * self.hp.per_patch_pattern_grid_shape[1],
            )
            max_winners_per_patch = math.ceil(max_winners / act.shape[0])
            winner_indices = (
                act.topk(max_winners_per_patch, dim=1)[1]
                + (torch.arange(0, act.shape[0]) * act.shape[1]).unsqueeze(dim=-1)
            ).view(-1)

            # convert winner indices back to image space
            winner_indices_i = torch.stack(
                [self.activation_index_signal_to_image(i) for i in winner_indices]
            )

        ## Other strategies for finding winners - listing for documentation
        ## sample based on activation
        # winner_indices = torch.multinomial(self.activation, 4)
        ## top N by activation
        # winner_indices = self.activation.topk(dim=0, k=4)[1]
        ## everone is a winner
        # winner_indices = torch.range(0, self.activation.nelement() - 1).long()

        return winner_indices_i
