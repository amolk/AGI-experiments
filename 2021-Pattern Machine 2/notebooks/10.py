# %% [markdown]
## Multiple patch high-d input to 2-d temporal-to-spatial map

# [Document]()

# %%
EXPERIMENT = "10"

from IPython import get_ipython
from IPython.display import display

print = display

import sys

sys.path.append("/Users/amolk/work/AGI/pattern-machine/src")

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("load_ext", "line_profiler")
get_ipython().run_line_magic("autoreload", "1")
get_ipython().run_line_magic("aimport", "patternmachine.layer")
get_ipython().run_line_magic("aimport", "patternmachine.signal_grid")
get_ipython().run_line_magic("aimport", "patternmachine.signal_grid_set")
get_ipython().run_line_magic("aimport", "patternmachine.signal_utils")
get_ipython().run_line_magic("aimport", "patternmachine.trace")
get_ipython().run_line_magic("aimport", "patternmachine.pattern_similarity")
get_ipython().run_line_magic("aimport", "patternmachine.pattern_grid")
get_ipython().run_line_magic("aimport", "patternmachine.clock")
get_ipython().run_line_magic("aimport", "patternmachine.utils")
get_ipython().run_line_magic("aimport", "patternmachine.input_output_patch_grid")
get_ipython().run_line_magic("aimport", "patternmachine.signal_source.video_source")
get_ipython().run_line_magic("matplotlib", "inline")

from typing import List, Tuple

# from skimage.transform import resize as resize_image
import cv2
from cv2 import resize as resize_image, trace
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
import numpy as np
import math
import pickle

from patternmachine.layer import Layer, LayerHP
from patternmachine.signal_grid import SignalGrid, SignalGridHP
from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.signal_source.moving_rectangle_signal_source import (
    MovingRectangleSignalSource,
)
from patternmachine.signal_source.rotating_rectangle_cached_signal_source import (
    RotatingRectangleCachedSignalSource,
)
from patternmachine.utils import show_image_grid
from patternmachine.similarity_utils import MIN_PRECISION


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=2)

TRAIN = True


class DynamicsExperiment1:
    def __init__(
        self,
        image_shape: Tuple = (25, 25),
        patch_grid_shape=(15, 15),
        per_patch_pattern_grid_shape: Tuple = (1, 1),
    ):
        self.image_shape = image_shape
        self.per_patch_pattern_grid_shape = per_patch_pattern_grid_shape

        # Layer
        self.layer_hp = self.create_layer_hp(
            image_shape=image_shape,
            input_coverage_factor=5,
            patch_grid_shape=patch_grid_shape,
            per_patch_pattern_grid_shape=per_patch_pattern_grid_shape,
            output_patch_neighborhood_shape=(5, 5),
            output_tau=0.3,
        )
        self.layer = Layer(hp=self.layer_hp)
        # self.layer.patterns.imshow()
        # self.signal_source = MovingRectangleSignalSource(
        #     height=image_shape[0], width=image_shape[1], step=1
        # )
        self.signal_source = RotatingRectangleCachedSignalSource(
            height=image_shape[0], width=image_shape[1], angle_step=5
        )

    def run(self, epochs=5, debug=True, learn=True, hallucinate_after=None):
        images = []
        images_vmin = []
        images_vmax = []
        t_end = 100  # self.signal_source.item_count
        avg_pattern_precision_mu = []
        avg_pattern_precision_output = []
        trace_input = None
        # image_size = (
        #     p_mu.hp.grid_shape[0] * p_mu.hp.grid_shape[2] * p_mu.signal_shape[0],
        #     p_mu.hp.grid_shape[1] * p_mu.hp.grid_shape[3] * p_mu.signal_shape[1],
        # )

        if debug:
            print(
                "mu, td_mu, patterns pixels, patterns precision, activation, output, handicap, mu_bar"
            )
        signal_source_items = self.signal_source.item()
        for epoch in tqdm(range(epochs)):
            # self.signal_source.seek(0)
            for t in range(1, t_end):
                avg_pattern_precision_mu.append(
                    self.layer.patterns.begin.components["mu"].precision.mean()
                )
                avg_pattern_precision_output.append(
                    self.layer.patterns.begin.components["__output__"].precision.mean()
                )

                raw_input_image = next(signal_source_items)

                if trace_input is None:
                    trace_input = raw_input_image.clone()
                    trace_input.components["mu"].pixels.epsilon = 0.5

                if hallucinate_after is not None and t > hallucinate_after and epoch > 0:
                    trace_input.components["mu"].pixels.data *= 0
                    trace_input.components["mu"].precision.data *= 0
                    trace_input.components["mu"].precision.data += MIN_PRECISION
                else:
                    trace_input = trace_input.trace_(raw_input_image)

                # if trace_input is None:
                #     trace_input = raw_input_image
                #     for component in trace_input.components.values():
                #         component.pixels.epsilon = 0.5
                # else:
                #     trace_input.trace_(raw_input_image)

                self.layer.forward(
                    trace_input=trace_input.clone(), learn=learn and (epoch > 0)
                )  # don't learn in first epoch to establish input trace

                if debug:
                    p_mu_image = self.layer.patterns.begin.components["mu"].pixels_as_image

                    mu_shape = raw_input_image.components["mu"].signal_shape

                    # raw_input
                    image = raw_input_image.components["mu"].pixels.view(mu_shape).numpy().copy()
                    image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                    images.append(image)
                    images_vmin.append(0)
                    images_vmax.append(1)

                    # mu
                    image = (self.layer.mu.components["mu"].pixels).view(mu_shape).numpy().copy()
                    image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                    images.append(image)
                    images_vmin.append(0)
                    images_vmax.append(1)

                    # mu precision
                    image = (
                        (self.layer.mu.components["mu"].precision).view(mu_shape).numpy().copy()
                    )
                    image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                    images.append(image)
                    images_vmin.append(0)
                    images_vmax.append(1)

                    # # pattern pixels BEGIN
                    # image = p_mu_image.numpy().copy()
                    # images.append(image)
                    # images_vmin.append(0)
                    # images_vmax.append(1)

                    # # pattern precisions BEGIN
                    # image = self.layer.patterns.begin.components["mu"].precision_as_image
                    # images.append(image.numpy().copy())
                    # images_vmin.append(0)
                    # images_vmax.append(1)

                    # # pattern pixels END
                    # image = self.layer.patterns.end.components["mu"].pixels_as_image
                    # images.append(image.numpy().copy())
                    # images_vmin.append(0)
                    # images_vmax.append(1)

                    # # pattern precisions END
                    # image = self.layer.patterns.end.components["mu"].precision_as_image
                    # images.append(image.numpy().copy())
                    # images_vmin.append(0)
                    # images_vmax.append(1)

                    # # activations
                    # image = (
                    #     self.layer.activation_i.view(self.layer.output.signal_shape).numpy().copy()
                    # )
                    # image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                    # images.append(image)
                    # images_vmin.append(0)
                    # images_vmax.append(None)

                    # raw_output
                    image = (
                        self.layer.raw_output.pixels.view(self.layer.output.signal_shape)
                        .numpy()
                        .copy()
                    )
                    image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                    images.append(image)
                    images_vmin.append(0)
                    images_vmax.append(None)

                    # output
                    image = (
                        self.layer.output.pixels.view(self.layer.output.signal_shape)
                        .numpy()
                        .copy()
                    )
                    image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                    images.append(image)
                    images_vmin.append(0)
                    images_vmax.append(None)

                    # handicap
                    image = (
                        self.layer.temporal_neighborhood_activation_handicap.view(
                            self.layer.output.signal_shape
                        )
                        .numpy()
                        .copy()
                    )
                    image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                    images.append(image)
                    images_vmin.append(None)
                    images_vmax.append(None)

                    # mu_bar
                    if self.layer.mu_bar is None:
                        image = np.zeros(mu_shape)
                    else:
                        image = (
                            self.layer.mu_bar.components["mu"].pixels.view(mu_shape).numpy().copy()
                        )
                    image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                    images.append(image)
                    images_vmin.append(None)
                    images_vmax.append(None)

                    # self.layer.patterns.begin.imshow()

                    # diff = (
                    #     self.mu.components["mu"].pixels[0, 0]
                    #     - self.layer.input_mu_bar.components["mu"].pixels[0, 0]
                    # )
                    # print("diff: ", diff.item())

            if epoch % math.ceil(epochs / 5) == 0:
                plt.plot(avg_pattern_precision_mu)
                plt.title("Avg pattern precision (mu)")
                plt.show()

                plt.plot(avg_pattern_precision_output)
                plt.title("Avg pattern precision (output)")
                plt.show()

            if debug:
                if epoch % math.ceil(epochs / 5) == 0 and (epoch > 0):
                    print("BEGIN")
                    self.layer.patterns.begin.components["mu"].imshow()
                    print("END")
                    self.layer.patterns.end.components["mu"].imshow()
                    show_image_grid(
                        images=np.stack(images),
                        grid_width=7,
                        grid_height=t_end - 1,
                        vmin=images_vmin,
                        vmax=images_vmax,
                    )
                    print("-----------------------")
                self.images = images
                images = []

    def make_mu(self, val, mu_shape):
        mu_signal_grid_hp = SignalGridHP(grid_shape=(1, 1), signal_shape=mu_shape)
        mu_signal_grid = SignalGrid(
            mu_signal_grid_hp, init_pixel_value=val, init_precision_value=1.0
        )

        mu = SignalGridSet.from_signal_grids({"mu": mu_signal_grid})
        return mu

    def set_pattern(self, pattern_index, val):
        self.layer.patterns.begin.components["mu"].pixels[pattern_index, 0] = val
        self.layer.patterns.begin.components["mu"].precision[pattern_index, 0] = 1.0
        self.layer.patterns.end.components["mu"].pixels[pattern_index, 0] = val
        self.layer.patterns.end.components["mu"].precision[pattern_index, 0] = 1.0

    def create_layer_hp(
        self,
        image_shape,
        input_coverage_factor=1.0,
        patch_grid_shape=(5, 5),
        per_patch_pattern_grid_shape=(1, 1),
        output_patch_neighborhood_shape=(1, 1),
        output_tau=1.0,
    ):
        hp = LayerHP(
            input_signal_shapes={"mu": image_shape},
            input_coverage_factor=input_coverage_factor,
            patch_grid_shape=patch_grid_shape,
            per_patch_pattern_grid_shape=per_patch_pattern_grid_shape,
            output_patch_neighborhood_shape=output_patch_neighborhood_shape,
            output_decay=output_tau,
            # pattern_init_pixel_values={"mu": 0.45, "__output__": 0.45},
            pattern_init_pixel_noise_amplitude={"mu": 0.1, "__output__": 0.1},
            pattern_init_precision_values={"mu": 0.01, "__output__": 0.01},
            # pattern_init_precision_noise_amplitude={"mu": 0.05, "__output__": 0.05},
            alpha=0.3,
        )  # set output_decay=1.0 for IID data

        hp.temporal_neighborhood_activation_handicap_weight = 0
        return hp


torch.set_printoptions(precision=4)
experiment = DynamicsExperiment1()

## PROFILE
# %prun -D prof experiment.run(epochs=100, debug=False)

if TRAIN:
    # TRAIN
    experiment.run(epochs=20, debug=True)
    # experiment.run(epochs=2, debug=True)
    # experiment.run(epochs=50, debug=False)
    # experiment.run(epochs=2, debug=True)
    # experiment.run(epochs=5, debug=False)
    with open(f"{EXPERIMENT}.patterns.pkl", "wb") as fp:
        pickle.dump(experiment.layer.patterns, fp)
    experiment.run(epochs=2, debug=True)
else:
    # LOAD MODEL
    with open(f"{EXPERIMENT}.patterns.pkl", "rb") as fp:
        experiment.layer.patterns = pickle.load(fp)
    experiment.run(epochs=2, debug=True, learn=False, hallucinate_after=None)

# %%
import os

import imageio
import matplotlib.pyplot as plt

print(len(experiment.images))
for img_index in [1, 3, 4, 6]:  # [1, 8, 10]:
    filenames = []
    for i in range(img_index, len(experiment.images), 7):
        filename = f"../outputs/{i}.png"
        plt.imshow(experiment.images[i])
        plt.savefig(filename)
        filenames.append(filename)

    # Build GIF
    with imageio.get_writer(f"./{EXPERIMENT}_{img_index}.gif", mode="I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)

# %%
# experiment.run(epochs=50, debug=False, learn=True, hallucinate_after=None)
experiment.run(epochs=2, debug=True, learn=False, hallucinate_after=None)
experiment.run(epochs=2, debug=True, learn=False, hallucinate_after=10)
# experiment.layer.patterns.begin.imshow()

# %%
%load_ext snakeviz
%snakeviz experiment.run(epochs=20, debug=False)
# on command line, run: snakeviz [path of profile file generated]
# %%
