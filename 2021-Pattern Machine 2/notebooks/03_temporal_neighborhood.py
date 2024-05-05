# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# constant input, single pre-learned pattern
# Single layer, input -> output -> input inh -> output inh -> input gain...
# Check if sinusoidal dynamics develop


# %%
import pdb
import sys

sys.path.append("/Users/amolk/work/AGI/pattern-machine/src")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "1")
from IPython.display import display

print = display

get_ipython().run_line_magic("aimport", "patternmachine.layer")
get_ipython().run_line_magic("aimport", "patternmachine.signal_grid_set")
get_ipython().run_line_magic("aimport", "patternmachine.signal_utils")
get_ipython().run_line_magic("aimport", "patternmachine.trace")
get_ipython().run_line_magic("aimport", "patternmachine.pattern_similarity")
get_ipython().run_line_magic("aimport", "patternmachine.pattern_grid")
get_ipython().run_line_magic("aimport", "patternmachine.clock")
get_ipython().run_line_magic("aimport", "patternmachine.utils")
get_ipython().run_line_magic("aimport", "patternmachine.input_output_patch_grid")


# %%
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import numpy as np
import math


def show_1d_image(image, title=None):
    assert len(image.shape) == 1
    fig = plt.figure(figsize=(2, 1))
    plt.axis("off")
    #     ax = fig.add_subplot(111)
    if title is not None:
        plt.title(label=title)
    plt.imshow(image.unsqueeze(0), vmin=0, vmax=1)
    #     ax.set_aspect('equal')
    #     plt.tight_layout(pad=0)
    plt.show()


def show_2d_image(image, title=None, text=False, vmin=0.0, vmax=1.0):
    assert len(image.shape) == 2
    if text:
        print(image)
    else:
        fig = plt.figure(figsize=(2, 2))
        plt.axis("off")
        if title is not None:
            plt.title(label=title)
        plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.show()


def show_image_grid(images, vmin=0, vmax=1, grid_width=None, grid_height=None):
    s = images.shape

    if type(vmin) is int:
        vmin = [vmin] * s[0]

    if type(vmax) is int:
        vmax = [vmax] * s[0]

    assert len(s) == 3
    if grid_width is None or grid_height is None:
        image_grid_size = math.floor(s[0] ** 0.5)
        if image_grid_size > 20:
            return
        grid_width = image_grid_size
        grid_height = image_grid_size
    else:
        assert grid_width * grid_height == s[0]

    image_height = s[1]
    image_width = s[2]
    image_aspect_ratio = image_height / image_width
    if grid_width == 1 and grid_height == 1:
        plt.figure(figsize=(grid_height, grid_width))
        plt.axis("off")
        plt.imshow(
            images[0].detach().cpu().numpy(),
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            cmap=plt.cm.viridis,
            aspect="auto",
        )
    else:
        # assert grid_width * grid_height == s[0], f"grid_width={grid_width}, h={grid_height}, grid_width*h={grid_width}*{grid_height}!={s[0]} number images"
        fig, axs = plt.subplots(
            nrows=grid_height,
            ncols=grid_width,
            figsize=(grid_width * 0.5, grid_height * 0.5 * image_aspect_ratio),
            subplot_kw={"xticks": [], "yticks": []},
        )

    axs = axs.flat
    for i in np.arange(grid_width * grid_height):
        axs[i].axis("off")
        axs[i].imshow(
            images[i].detach().cpu().numpy(),
            vmin=vmin[i],
            vmax=vmax[i],
            interpolation="none",
            cmap=plt.cm.viridis,
            aspect="auto",
        )

    fig.subplots_adjust(top=1, left=0, bottom=0, right=1, wspace=0.1, hspace=0.1)

    plt.show()


# %%
# %matplotlib ipympl

from typing import Tuple

import matplotlib.pyplot as plt
import torch

# from tqdm.notebook import tqdm
from tqdm import tqdm

from patternmachine.layer import Layer, LayerHP
from patternmachine.signal_grid import SignalGridHP, SignalGrid
from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.signal_utils import SignalUtils
from patternmachine.input_output_patch_grid import InputOutputPatchGrid
from patternmachine.pattern_similarity import PatternSimilarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DynamicsExperiment1:
    def __init__(self, mu_shape: Tuple = (1, 1), pattern_grid_shape: Tuple = (5, 5)):
        self.mu_shape = mu_shape
        self.pattern_grid_shape = pattern_grid_shape

        # Layer
        self.layer_hp = self.create_layer_hp(per_patch_pattern_grid_shape=pattern_grid_shape)
        self.layer = Layer(hp=self.layer_hp)

        self.mu = self.make_mu(val=0.3, mu_shape=mu_shape)
        self.set_pattern(pattern_index=12, val=0.3)
        self.mu2 = self.make_mu(val=0.9, mu_shape=mu_shape)
        self.set_pattern(pattern_index=11, val=0.9)
        self.set_pattern(pattern_index=0, val=0.9)

        output_values = []
        images = []
        images_vmin = []
        images_vmax = []
        t_end = 100
        for t in range(t_end):
            # print("Time step ", t)
            # image = self.layer.next_handicap.view(self.layer.output.signal_shape)
            # image = self.layer.temporal_neighborhood_activation_handicap.pixels.view(
            #     self.layer.output.signal_shape
            # )

            # output_values.append(self.layer.output.pixels.squeeze(dim=0))
            output_values.append(self.layer.activation)
            images.append(self.layer.activation.view(self.layer.output.signal_shape))
            images_vmin.append(0)
            images_vmax.append(1)

            images.append(self.layer.output.pixels.view(self.layer.output.signal_shape))
            images_vmin.append(0)
            images_vmax.append(1)

            images.append(self.layer.next_handicap.view(self.layer.output.signal_shape))
            images_vmin.append(0)
            images_vmax.append(1.2)

            images.append(
                self.layer.temporal_neighborhood_activation_handicap.pixels.view(
                    self.layer.output.signal_shape
                )
            )
            images_vmin.append(0)
            images_vmax.append(1.2)

            if t % 2 == 0:
                self.layer.forward(input_s=self.mu)
            else:
                self.layer.forward(input_s=self.mu2)

        show_image_grid(
            images=torch.stack(images),
            grid_width=4,
            grid_height=t_end,
            vmin=images_vmin,
            vmax=images_vmax,
        )
        output_values = torch.stack(output_values)[:, (12, 11, 0)].transpose(0, 1)
        labels = ["pattern 0.3", "close pattern 0.9", "distant pattern 0.9"]
        x = range(0, output_values.shape[-1])
        for v, name in zip(output_values, labels):
            plt.plot(x, v, label=name)
        plt.legend()
        plt.show()

    def make_mu(self, val, mu_shape):
        mu_signal_grid_hp = SignalGridHP(grid_shape=(1, 1), signal_shape=mu_shape)
        mu_signal_grid = SignalGrid(
            mu_signal_grid_hp, init_pixel_value=val, init_precision_value=1.0
        )

        mu = SignalGridSet.from_signal_grids({"mu": mu_signal_grid})
        return mu

    def set_pattern(self, pattern_index, val):
        self.layer.patterns.begin.components["mu"].pixels[pattern_index, 0] = val
        self.layer.patterns.end.components["mu"].pixels[pattern_index, 0] = val

    def create_layer_hp(
        self,
        input_coverage_factor=1.0,
        patch_grid_shape=(1, 1),
        per_patch_pattern_grid_shape=(1, 1),
        output_patch_neighborhood_shape=(1, 1),
        output_tau=1.0,
    ):
        hp = LayerHP(
            input_signal_shapes={"mu": self.mu_shape},
            input_coverage_factor=input_coverage_factor,
            patch_grid_shape=patch_grid_shape,
            per_patch_pattern_grid_shape=per_patch_pattern_grid_shape,
            output_patch_neighborhood_shape=output_patch_neighborhood_shape,
            output_decay=output_tau,
            pattern_init_pixel_values={"mu": 0, "__output__": 0},
            pattern_init_precision_values={"mu": 1, "__output__": 1},
        )  # set output_decay=1.0 for IID data
        return hp


experiment = DynamicsExperiment1()

# %%
