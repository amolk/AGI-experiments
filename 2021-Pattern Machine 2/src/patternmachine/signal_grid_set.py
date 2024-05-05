from typing import Dict, List, Tuple
import copy

import torch
from config import Config
import matplotlib.pyplot as plt
from patternmachine.conv_utils import ImagePatchesInfo

from patternmachine.signal_grid import SignalGrid, SignalGridHP
from patternmachine.utils import (
    make_2d_image,
    make_2d_image_alpha,
    pretty_s,
    show_2d_image,
    show_2d_image_alpha,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GridShapeMismatchError(Exception):
    pass


class NoComponentsError(Exception):
    pass


class SignalGridSetHP:
    def __init__(self, hps: Dict[str, SignalGridHP]):
        if len(hps) == 0:
            raise NoComponentsError("Must specify at least one component")

        self.components = hps
        self.grid_shape = hps[list(hps.keys())[0]].grid_shape

        # all components must have same grid size
        for _, component_hp in hps.items():
            if component_hp.grid_shape != self.grid_shape:
                raise GridShapeMismatchError


class SignalGridSet:
    @staticmethod
    def from_pixels_list(pixels_list, signal_shape=None) -> "SignalGridSet":
        if signal_shape is None:
            signal_shape = {name: pixels.shape for name, pixels in pixels_list.items()}
        signal_hps = {
            name: SignalGridHP(grid_shape=(1, 1), signal_shape=signal_shape[name])
            for name, pixels in pixels_list.items()
        }
        signal_grids = {
            name: SignalGrid(hp=signal_hp, pixels=pixels_list[name].view(1, -1))
            for name, signal_hp in signal_hps.items()
        }
        return SignalGridSet.from_signal_grids(signal_grids)

    @staticmethod
    def from_pixels_and_precisions_list(
        pixels_list: List[torch.Tensor],
        precisions_list: List[torch.Tensor],
        signal_shape: List[Tuple],
    ):
        if signal_shape is None:
            signal_shape = {name: pixels.shape for name, pixels in pixels_list.items()}
        signal_hps = {
            name: SignalGridHP(grid_shape=(1, 1), signal_shape=signal_shape[name])
            for name, pixels in pixels_list.items()
        }
        signal_grids = {
            name: SignalGrid(
                hp=signal_hp, pixels=pixels_list[name], precision=precisions_list[name]
            )
            for name, signal_hp in signal_hps.items()
        }
        return SignalGridSet.from_signal_grids(signal_grids)

    @staticmethod
    def from_signal_grids(signal_grids: Dict[str, SignalGrid]):
        result = SignalGridSet(hp=None)
        hp = SignalGridSetHP(hps={name: sg.hp for name, sg in signal_grids.items()})
        result.hp = hp
        result.components = signal_grids
        return result

    @staticmethod
    def hallucinate_with_attention(sgs_list):
        result = SignalGridSet(hp=sgs_list[0].hp, alloc=False)
        components = {}
        for component_name in result.components:
            # print("Component", component_name)
            # print("sgs1", sgs1.components[component_name])
            # print("sgs2", sgs2.components[component_name])
            l = [
                sgs.components[component_name]
                for sgs in sgs_list
                if component_name in sgs.components
            ]

            if len(l) == 2:
                component = SignalGrid.hallucinate_with_attention(l)
                components[component_name] = component
        result.components = components
        return result

    @staticmethod
    def precision_weighted_mix(sgs_list):
        result = SignalGridSet(hp=sgs_list[0].hp, alloc=False)
        components = {}
        for component_name in result.components:
            # print("Component", component_name)
            # print("sgs1", sgs1.components[component_name])
            # print("sgs2", sgs2.components[component_name])
            component = SignalGrid.precision_weighted_mix(
                [
                    sgs.components[component_name]
                    for sgs in sgs_list
                    if component_name in sgs.components
                ]
            )
            components[component_name] = component
        result.components = components
        return result

    @staticmethod
    def temporal_diff(current: "SignalGridSet", previous: "SignalGridSet"):
        result: SignalGridSet = current - previous

        # result is in range [-1,1]
        # we need to tranform it into [0, 1], such that 0 becomes 0.3
        for component_name in result.components:
            component: SignalGrid = result.components[component_name]

            # [-1, 1] -> [-0.7, 0.7]
            component.pixels *= 1 - Config.BASE_ACTIVATION

            # [-0.7, 0.7] -> [-0.4, 1]
            component.pixels += Config.BASE_ACTIVATION

            # [-0.4, 1.0] -> [0, 1]
            component.pixels.clamp_(min=0.0, max=1.0)

        return result

    def __init__(
        self,
        hp: SignalGridSetHP = None,
        alloc=True,
        init_pixel_values: Dict = None,
        init_pixel_kernels: Dict = None,
        init_pixel_noise_amplitude: Dict = None,
        init_precision_values: Dict = None,
        init_precision_kernels: Dict = None,
        init_precision_noise_amplitude: Dict = None,
    ):
        self.hp = hp
        if hp:
            if init_pixel_values is None:
                init_pixel_values = {name: None for name in hp.components.keys()}

            if init_pixel_kernels is None:
                init_pixel_kernels = {name: None for name in hp.components.keys()}

            if init_pixel_noise_amplitude is None:
                init_pixel_noise_amplitude = {name: 0.0 for name in hp.components.keys()}

            if init_precision_values is None:
                init_precision_values = {name: None for name in hp.components.keys()}

            if init_precision_kernels is None:
                init_precision_kernels = {name: None for name in hp.components.keys()}

            if init_precision_noise_amplitude is None:
                init_precision_noise_amplitude = {name: 0.0 for name in hp.components.keys()}

            self.components = {
                name: SignalGrid(
                    component_hp,
                    alloc_pixels=alloc,
                    init_pixel_value=init_pixel_values[name],
                    init_pixel_kernel=init_pixel_kernels[name],
                    init_pixel_noise_amplitude=init_pixel_noise_amplitude[name],
                    init_precision_value=init_precision_values[name],
                    init_precision_kernel=init_precision_kernels[name],
                    init_precision_noise_amplitude=init_precision_noise_amplitude[name],
                )
                for name, component_hp in hp.components.items()
            }

    # add a SignalGrid as a component
    def add_component(self, name: str, signal_grid: SignalGrid):
        assert (
            self.hp.grid_shape == signal_grid.hp.grid_shape
        ), f"{self.hp.grid_shape} != {signal_grid.hp.grid_shape}"

        self.hp.components[name] = signal_grid.hp
        self.components[name] = signal_grid

    @property
    def signal_shape(self):
        return {name: c.hp.signal_shape for name, c in self.components.items()}

    @property
    def pixels(self):
        return {name: component.pixels for name, component in self.components.items()}

    @property
    def precision(self):
        return {name: component.precision for name, component in self.components.items()}

    @property
    def pixels_as_image(self):
        return {name: component.pixels_as_image for name, component in self.components.items()}

    @property
    def precision_as_image(self):
        return {name: component.precision_as_image for name, component in self.components.items()}

    @property
    def grid_shape(self):
        return self.hp.grid_shape

    @property
    def component_count(self):
        return len(self.components)

    @property
    def pixel_contrast(self):
        return {name: component.pixel_contrast for name, component in self.components.items()}

    @property
    def patches_info(self):
        return {name: component.patches_info for name, component in self.components.items()}

    def __getitem__(self, index):
        pixels_list = {
            name: component.pixels[index] for name, component in self.components.items()
        }

        return SignalGridSet.from_pixels_list(pixels_list, self.signal_shape)

    def __sub__(self, other: "SignalGridSet"):
        return self._op(operation="__sub__", other=other)

    def __add__(self, other: "SignalGridSet"):
        return self._op(operation="__add__", other=other)

    def trace_(self, other):
        return self._op(operation="trace_", other=other)

    def clone(self):
        return self._self_op(operation="clone")

    def _op(self, operation, other: "SignalGridSet"):
        assert self.hp.grid_shape == other.hp.grid_shape
        assert self.component_count == other.component_count
        result = SignalGridSet(hp=copy.deepcopy(self.hp), alloc=False)
        result_components = {}
        for component_name in self.components:
            component = self.components[component_name]
            other_component = other.components[component_name]
            assert component.signal_shape == other_component.signal_shape
            if operation == "__sub__":
                result_components[component_name] = component - other_component
            elif operation == "__add__":
                result_components[component_name] = component + other_component
            elif operation == "trace_":
                result_components[component_name] = component
                result_components[component_name].trace_(other_component)

        result.components = result_components
        return result

    def _self_op(self, operation):
        result = SignalGridSet(hp=self.hp, alloc=False)
        result_components = {}
        for component_name in self.components:
            component = self.components[component_name]
            if operation == "clone":
                result_components[component_name] = component.clone()

        result.components = result_components
        return result

    def __repr__(self):
        return pretty_s("", self)

    def weighted_mixture(
        self,
        weights: torch.Tensor,
        patches_info: Dict[str, ImagePatchesInfo],
        output_sgshp: SignalGridSetHP,
    ):
        assert len(self.grid_shape) == 4, "Implemented only for pattern grids"

        mixed_signals = {}
        for component_name in output_sgshp.components:
            component = self.components[component_name]
            mixed_signals[component_name] = component.weighted_mixture(
                patches_info=patches_info[component_name],
                weights=weights,
                output_sghp=output_sgshp.components[component_name],
            )

        return SignalGridSet.from_signal_grids(mixed_signals)

    def imshow(self, title=None):
        for component_name in self.components:
            if title is None:
                component_title = component_name
            else:
                component_title = f"{title} - {component_name}"
            component = self.components[component_name]

            component.imshow(title=component_name)