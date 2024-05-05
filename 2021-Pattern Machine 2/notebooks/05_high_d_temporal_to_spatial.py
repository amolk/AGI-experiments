# %% [markdown]
## Temporal signal in input space to spatial signal in pattern space

# [Document](https://docs.google.com/document/d/1Dy-IpKhKZ3rgHPBC58whxNGZ9hQW2ITaYaPuvJ1dl_k/edit#)

# %%
from IPython import get_ipython

# %%
# constant input, single pre-learned pattern
# Single layer, input -> output -> input inh -> output inh -> input gain...
# Check if sinusoidal dynamics develop


# %%
import pdb
import sys

from scipy.ndimage import interpolation

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

    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

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
            images[0],
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
            images[i],
            vmin=vmin[i],
            vmax=vmax[i],
            interpolation="none",
            cmap=plt.cm.viridis,
            aspect="auto",
        )

    fig.subplots_adjust(top=1, left=0, bottom=0, right=1, wspace=0.1, hspace=0.1)

    plt.show()


def create_line_image(size, angle, color=1, thickness=1, blur_kernel=(3, 3)):
    original_size = size
    size = size * 2
    img = np.zeros((size, size))
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    cv2.line(
        img,
        pt1=(0, int(image_center[1])),
        pt2=(size - 1, int(image_center[1])),
        color=color,
        thickness=thickness,
    )
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    img = cv2.blur(img, blur_kernel)
    img = img[
        int(original_size / 2) + 1 : int(original_size / 2) + original_size + 1,
        int(original_size / 2) + 1 : int(original_size / 2) + original_size + 1,
    ]
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()
    return torch.tensor(img).float()


def create_rotated_rect_image(size, angle, color=1, thickness=-1):
    original_size = size
    size = size * 2
    img = np.zeros((size, size))
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    cv2.rectangle(
        img,
        pt1=(0, 0),
        pt2=(size - 1, int(image_center[1])),
        color=color,
        thickness=thickness,
    )
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    img = img[
        int(original_size / 2) + 1 : int(original_size / 2) + original_size + 1,
        int(original_size / 2) + 1 : int(original_size / 2) + original_size + 1,
    ]
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()
    return torch.tensor(img).float()


def temporal_diff(image_t0, image_t1):
    img = image_t1 - image_t0
    img = (img + 0.6) / 2.0
    if type(img) is np.ndarray:
        img = img.clip(min=0, max=1)
    else:
        img.clamp_(min=0, max=1)
    # plt.imshow(img)
    return img


# %%
# %matplotlib ipympl

from typing import Tuple

import matplotlib.pyplot as plt
import torch

# from skimage.transform import resize as resize_image
import cv2
from cv2 import resize as resize_image

# from tqdm.notebook import tqdm
from tqdm import tqdm

from patternmachine.layer import Layer, LayerHP
from patternmachine.signal_grid import SignalGridHP, SignalGrid
from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.signal_utils import SignalUtils
from patternmachine.input_output_patch_grid import InputOutputPatchGrid
from patternmachine.pattern_similarity import PatternSimilarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=2)


class DynamicsExperiment1:
    def __init__(self, mu_shape: Tuple = (11, 11), pattern_grid_shape: Tuple = (7, 7)):
        self.mu_shape = mu_shape
        self.pattern_grid_shape = pattern_grid_shape

        # Layer
        self.layer_hp = self.create_layer_hp(per_patch_pattern_grid_shape=pattern_grid_shape)
        self.layer = Layer(hp=self.layer_hp)

        self.mus: List[SignalGridSet] = []
        start_angle = -180
        end_angle = 180
        angle_step = 10
        mu = self.make_mu_from_image(
            create_rotated_rect_image(size=mu_shape[0], angle=start_angle - angle_step)
            # create_line_image(size=mu_shape[0], angle=start_angle - angle_step)
        )
        for angle in range(start_angle, end_angle, angle_step):
            next_mu = self.make_mu_from_image(
                create_rotated_rect_image(size=mu_shape[0], angle=angle)
            )
            # next_mu = self.make_mu_from_image(create_line_image(size=mu_shape[0], angle=angle))
            mu.trace_(next_mu)
            self.mus.append(mu.clone())

        self.mus = self.mus[int(len(self.mus) / 2) :]
        print("Motion blurred inputs")
        for mu in self.mus:
            plt.imshow(mu.pixels_as_image["mu"])
            plt.show()

        # for i, val in enumerate(np.linspace(0, 1, num=5)):
        #     self.set_pattern(pattern_index=i * 6, val=val)

        images = []
        images_vmin = []
        images_vmax = []
        epochs = 50
        t_end = len(self.mus)
        # image_size = (
        #     p_mu.hp.grid_shape[0] * p_mu.hp.grid_shape[2] * p_mu.signal_shape[0],
        #     p_mu.hp.grid_shape[1] * p_mu.hp.grid_shape[3] * p_mu.signal_shape[1],
        # )

        print("mu, patterns pixels, patterns precision, activation, output, handicap, mu_bar")
        for epoch in range(epochs):
            for t in range(1, t_end):
                mu = self.mus[t % len(self.mus)]
                mu_prev = self.mus[(t - 1) % len(self.mus)]

                self.mu = SignalGridSet.temporal_diff(current=mu, previous=mu_prev)
                # print("-" * 10)
                # print(f"input {self.mu.components['mu'].pixels[0, 0].item()}")

                self.layer.forward(input_s=self.mu)
                p_mu_image = self.layer.patterns.begin.components["mu"].pixels_as_image

                # mu
                image = self.mu.components["mu"].pixels.view(mu_shape).numpy().copy()
                image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                images.append(image)
                images_vmin.append(0)
                images_vmax.append(1)

                # pattern pixels
                image = p_mu_image.numpy().copy()
                images.append(image)
                images_vmin.append(0)
                images_vmax.append(1)

                # pattern precisions
                image = self.layer.patterns.begin.components["mu"].precision_as_image
                images.append(image.numpy().copy())
                images_vmin.append(0)
                images_vmax.append(1)

                # activations
                image = self.layer.activation.view(self.layer.output.signal_shape).numpy().copy()
                image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                images.append(image)
                images_vmin.append(0)
                images_vmax.append(1)

                # output
                image = (
                    self.layer.output.pixels.view(self.layer.output.signal_shape).numpy().copy()
                )
                image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                images.append(image)
                images_vmin.append(0)
                images_vmax.append(1)

                # images.append(
                #     self.layer.next_handicap.view(self.layer.output.signal_shape).numpy().copy()
                # )
                # images_vmin.append(0)
                # images_vmax.append(1.2)

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
                images_vmin.append(-0.5)
                images_vmax.append(1.5)

                # mu_bar
                image = (
                    self.layer.input_mu_bar.components["mu"].pixels.view(mu_shape).numpy().copy()
                )
                image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                images.append(image)
                images_vmin.append(0)
                images_vmax.append(1)

                # diff = (
                #     self.mu.components["mu"].pixels[0, 0]
                #     - self.layer.input_mu_bar.components["mu"].pixels[0, 0]
                # )
                # print("diff: ", diff.item())

            if epoch % int(epochs / 5) == 0:
                show_image_grid(
                    images=np.stack(images),
                    grid_width=7,
                    grid_height=t_end - 1,
                    vmin=images_vmin,
                    vmax=images_vmax,
                )
            self.images = images
            images = []

    def make_mu(self, val, mu_shape):
        mu_signal_grid_hp = SignalGridHP(grid_shape=(1, 1), signal_shape=mu_shape)
        mu_signal_grid = SignalGrid(
            mu_signal_grid_hp, init_pixel_value=val, init_precision_value=1.0
        )

        mu = SignalGridSet.from_signal_grids({"mu": mu_signal_grid})
        return mu

    def make_mu_from_image(self, image):
        mu_signal_grid_hp = SignalGridHP(grid_shape=(1, 1), signal_shape=image.shape)
        mu_signal_grid = SignalGrid(
            mu_signal_grid_hp,
            alloc_pixels=False,
            pixels=image.view(-1).unsqueeze(0),
            init_precision_value=1.0,
        )
        mu_signal_grid.pixels.epsilon = 0.5
        mu_signal_grid.precision.epsilon = 0.5
        mu = SignalGridSet.from_signal_grids({"mu": mu_signal_grid})
        return mu

    def set_pattern(self, pattern_index, val):
        self.layer.patterns.begin.components["mu"].pixels[pattern_index, 0] = val
        self.layer.patterns.begin.components["mu"].precision[pattern_index, 0] = 1.0
        self.layer.patterns.end.components["mu"].pixels[pattern_index, 0] = val
        self.layer.patterns.end.components["mu"].precision[pattern_index, 0] = 1.0

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
            # pattern_init_pixel_values={"mu": 0.45, "__output__": 0.45},
            pattern_init_precision_values={"mu": 0.01, "__output__": 0.01},
        )  # set output_decay=1.0 for IID data
        return hp


experiment = DynamicsExperiment1()

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

print(len(experiment.images))
filenames = []
for i in range(4, len(experiment.images), 7):
    filename = f"../outputs/{i}.png"
    plt.imshow(experiment.images[i])
    plt.savefig(filename)
    filenames.append(filename)

# Build GIF
with imageio.get_writer("./05_high_d_temporal_to_spatial.gif", mode="I") as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in filenames:
    os.remove(filename)
