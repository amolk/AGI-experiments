# %% [markdown]
## Multiple patch high-d input to 2-d temporal-to-spatial map

# [Document]()

# %%
from IPython import get_ipython

import pdb
import sys

from scipy.ndimage import interpolation

sys.path.append("/Users/amolk/work/AGI/pattern-machine/src")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "1")
from IPython.display import display

print = display

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

    if type(vmin) is int or vmin is None:
        vmin = [vmin] * s[0]

    if type(vmax) is int or vmax is None:
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
            figsize=(grid_width * 2, grid_height * image_aspect_ratio * 2),
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

from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

# from skimage.transform import resize as resize_image
import cv2
from cv2 import resize as resize_image

from tqdm.notebook import tqdm

# from tqdm import tqdm

from patternmachine.layer import Layer, LayerHP
from patternmachine.signal_grid import SignalGridHP, SignalGrid
from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.signal_utils import SignalUtils
from patternmachine.input_output_patch_grid import InputOutputPatchGrid
from patternmachine.pattern_similarity import PatternSimilarity
from patternmachine.similarity_utils import MAX_PRECISION, MIN_PRECISION
from config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=2)


class DynamicsExperiment1:
    def __init__(
        self,
        image_shape: Tuple = (23, 23),
        patch_grid_shape=(5, 5),
        per_patch_pattern_grid_shape: Tuple = (11, 11),
    ):
        self.image_shape = image_shape
        self.per_patch_pattern_grid_shape = per_patch_pattern_grid_shape

        # Layer
        self.layer_hp = self.create_layer_hp(
            image_shape=image_shape,
            input_coverage_factor=2.2,
            patch_grid_shape=patch_grid_shape,
            per_patch_pattern_grid_shape=per_patch_pattern_grid_shape,
            output_patch_neighborhood_shape=(1, 1),
            output_tau=0.4,
        )
        self.layer = Layer(hp=self.layer_hp)
        self.layer.patterns.imshow()

        self.raw_input_images: List[SignalGridSet] = []
        start_angle = 0
        end_angle = 720
        angle_step = 10
        sgs = self.make_sgs_from_image(
            create_rotated_rect_image(size=image_shape[0], angle=start_angle - angle_step)
            # create_line_image(size=image_shape[0], angle=start_angle - angle_step)
        )
        for angle in range(start_angle, end_angle, angle_step):
            next_sgs = self.make_sgs_from_image(
                create_rotated_rect_image(size=image_shape[0], angle=angle)
            )
            # next_sgs = self.make_sgs_from_image(create_line_image(size=mu_shape[0], angle=angle))
            # next_sgs = next_sgs * (1 - Config.BASE_ACTIVATION) + Config.BASE_ACTIVATION
            sgs.trace_(next_sgs)

            self.raw_input_images.append(sgs.clone())

        self.raw_input_images = self.raw_input_images[int(len(self.raw_input_images) / 2) :]
        # print("Motion blurred inputs")
        # for sgs in self.raw_input_images:
        #     plt.imshow(sgs.pixels_as_image["mu"])
        #     plt.show()

        # for i, val in enumerate(np.linspace(0, 1, num=5)):
        #     self.set_pattern(pattern_index=i * 6, val=val)

    def run(self, epochs=5):
        images = []
        images_vmin = []
        images_vmax = []
        t_end = len(self.raw_input_images)
        avg_pattern_precision_mu = []
        avg_pattern_precision_output = []
        # image_size = (
        #     p_mu.hp.grid_shape[0] * p_mu.hp.grid_shape[2] * p_mu.signal_shape[0],
        #     p_mu.hp.grid_shape[1] * p_mu.hp.grid_shape[3] * p_mu.signal_shape[1],
        # )

        print(
            "mu, td_mu, patterns pixels, patterns precision, activation, output, handicap, mu_bar"
        )
        for epoch in tqdm(range(epochs)):
            for t in range(1, t_end):
                avg_pattern_precision_mu.append(
                    self.layer.patterns.begin.components["mu"].precision.mean()
                )
                avg_pattern_precision_output.append(
                    self.layer.patterns.begin.components["__output__"].precision.mean()
                )

                raw_input_image = self.raw_input_images[t % len(self.raw_input_images)]
                self.layer.forward(trace_input=raw_input_image)
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
                image = (self.layer.mu.components["mu"].precision).view(mu_shape).numpy().copy()
                image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                images.append(image)
                images_vmin.append(0)
                images_vmax.append(1)

                # pattern pixels BEGIN
                image = p_mu_image.numpy().copy()
                images.append(image)
                images_vmin.append(0)
                images_vmax.append(1)

                # pattern precisions BEGIN
                image = self.layer.patterns.begin.components["mu"].precision_as_image
                images.append(image.numpy().copy())
                images_vmin.append(0)
                images_vmax.append(1)

                # pattern pixels END
                image = self.layer.patterns.end.components["mu"].pixels_as_image
                images.append(image.numpy().copy())
                images_vmin.append(0)
                images_vmax.append(1)

                # pattern precisions END
                image = self.layer.patterns.end.components["mu"].precision_as_image
                images.append(image.numpy().copy())
                images_vmin.append(0)
                images_vmax.append(1)

                # activations
                image = self.layer.activation_i.view(self.layer.output.signal_shape).numpy().copy()
                image = resize_image(image, p_mu_image.shape, interpolation=cv2.INTER_NEAREST)
                images.append(image)
                images_vmin.append(0)
                images_vmax.append(None)

                # output
                image = (
                    self.layer.output.pixels.view(self.layer.output.signal_shape).numpy().copy()
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
                    image = self.layer.mu_bar.components["mu"].pixels.view(mu_shape).numpy().copy()
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

                print("BEGIN")
                self.layer.patterns.begin.components["mu"].imshow()
                print("END")
                self.layer.patterns.end.components["mu"].imshow()
                show_image_grid(
                    images=np.stack(images),
                    grid_width=11,
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

    def make_sgs_from_image(self, image):
        mu_signal_grid_hp = SignalGridHP(grid_shape=(1, 1), signal_shape=image.shape)
        mu_signal_grid = SignalGrid(
            mu_signal_grid_hp,
            alloc_pixels=False,
            pixels=image.view(-1).unsqueeze(0),
            init_precision_value=MAX_PRECISION,
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
            alpha=0.1,
        )  # set output_decay=1.0 for IID data

        hp.temporal_neighborhood_activation_handicap_weight = 1.0
        return hp


torch.set_printoptions(precision=4)
experiment = DynamicsExperiment1()
experiment.run(epochs=100)

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

print(len(experiment.images))
for img_index in [8, 10]:
    filenames = []
    for i in range(img_index, len(experiment.images), 11):
        filename = f"../outputs/{i}.png"
        plt.imshow(experiment.images[i])
        plt.savefig(filename)
        filenames.append(filename)

    # Build GIF
    with imageio.get_writer(f"./07_{img_index}.gif", mode="I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)

# %%
# %prun -D prof experiment.run(epochs=10)
experiment.run(epochs=100)
# experiment.layer.patterns.begin.imshow()

# %%
