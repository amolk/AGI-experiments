import math
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d

from .config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPSILON = 0.00001
KERNEL_RANDOM = "random"
KERNEL_GAUSSIAN = "gaussian"
KERNEL_DOG = "dog"
VP_FACTOR = 3


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


def make_2d_image(image, axis, title=None, vmin=None, vmax=None):
    if title is not None:
        axis.set_title(title)
    axis.imshow(image, vmin=vmin, vmax=vmax)
    axis.set_axis_off()


def show_2d_image(image, title=None, vmin=None, vmax=None):
    assert len(image.shape) == 2
    if title:
        print(title)
    fig, ax = plt.subplots(figsize=(image.shape[0] / 20, image.shape[1] / 20))
    make_2d_image(image, ax, title, vmin, vmax)
    plt.show()


def make_2d_image_alpha(image, axis, alpha, title=None):
    axis.imshow(torch.ones_like(image) * 0.2, vmin=0, vmax=1, cmap="RdGy")
    axis.imshow(image, alpha=alpha, vmin=0, vmax=1, cmap="gray")
    axis.set_axis_off()
    if title is not None:
        axis.set_title(title)


def show_2d_image_alpha(image, alpha, title=None):
    if title:
        print(title)
    fig, ax = plt.subplots(figsize=(image.shape[0] / 20, image.shape[1] / 20))
    make_2d_image_alpha(image, ax, alpha, title)
    plt.show()


def show_image_grid(
    images, vmin=0, vmax=1, grid_width=None, grid_height=None, filename=None
):
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

    if filename:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()
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


def pretty_s(name, klass, indent=0):
    if type(klass).__name__ in ["Tensor", "Trace"]:
        return (
            " " * indent
            + name
            + ":"
            + type(klass).__name__
            + " size"
            + str(tuple(klass.shape))
        )

    strs = [" " * indent + name + ":" + type(klass).__name__]

    indent += 2
    if hasattr(klass, "__info__"):
        info = klass.__info__
    else:
        info = klass.__dict__

    for k, v in info.items():
        if hasattr(v, "__info__") or hasattr(v, "__dict__"):
            strs.append(pretty_s(k, v, indent))
        elif "__iter__" in dir(v):
            if type(v) is tuple:
                strs.append(" " * indent + k + " = " + str(v))
            else:
                strs.append(" " * indent + k + " = [")
                for index, item in enumerate(v):
                    if hasattr(item, "__info__") or hasattr(item, "__dict__"):
                        strs.append(pretty_s(str(index), item, indent + 2))
                    else:
                        strs.append(" " * (indent + 2) + str(item))
                strs.append(" " * indent + "]")
        else:
            strs.append(" " * indent + k + " = " + str(v))

    return "\n".join(strs)


def pretty_print(name, clas, indent=0):
    print(pretty_s(name, clas, indent))


def soft_add(a, b, tau):
    return a * (1 - tau) + b * tau


def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    t = tensor + torch.randn(tensor.size()).to(device) * std + mean
    t.to(device)
    return t


def plot_patterns(
    patterns,
    pattern_lr,
    dataset,
    voronoi=False,
    annotate=False,
    figsize=(7, 7),
    dpi=100,
):
    patterns = patterns.cpu()
    dataset = dataset.cpu()
    assert len(patterns.shape) == 2  # (pattern count, 2)
    assert patterns.shape[1] == 2  # 2D

    rgba_colors = torch.zeros((patterns.shape[0], 4))

    # for blue the last column needs to be one
    rgba_colors[:, 2] = 1.0
    # the fourth column needs to be your alphas
    if pattern_lr is not None:
        alpha = (1.1 - pattern_lr.cpu()).clamp(0, 1) * 0.9
        rgba_colors[:, 3] = alpha
    else:
        rgba_colors[:, 3] = 1.0

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    ax.cla()  # clear things for fresh plot

    if annotate:
        for i in range(patterns.shape[0]):
            ax.annotate(
                str(i),
                (patterns[i][0], patterns[i][1]),
                xytext=(5, -3),
                textcoords="offset points",
            )

    ax.scatter(patterns[:, 0], patterns[:, 1], marker=".", c=rgba_colors, s=50)
    ax.scatter(dataset[:, 0], dataset[:, 1], marker=".", c="r", s=10)

    if voronoi:
        vor = Voronoi(patterns)
        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            line_colors="gray",
            line_width=1,
            line_alpha=0.2,
            point_size=0,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()


"""
Create a numpy array of given shape, each element initialized using supplied function
Example: make_ndarray((2,3), lambda multi_index:multi_index)
"""


def make_ndarray(shape, fn):
    a = np.empty(shape, dtype=object)
    with np.nditer(a, flags=["refs_ok", "multi_index"], op_flags=["readwrite"]) as it:
        for _ in it:
            a[it.multi_index] = fn(it.multi_index)

    return a


def sigmoid_addition(stacked_tensors, dim=0):
    # c = torch.logit(torch.clamp_min(a, min=EPSILON)) + torch.logit(torch.clamp_min(b, min=EPSILON))
    stacked_tensors = torch.logit(torch.clamp_min(stacked_tensors, min=EPSILON))
    c = torch.sum(stacked_tensors, dim=dim)

    return torch.sigmoid(c)


def inverse(x, scale=7):
    """map [0,1] to [1,0] using reflected sigmoid function"""
    x = torch.clamp(x, min=0, max=1)
    return 1 / (1 + torch.exp(scale * (2 * x - 1)))


def normalize(x, dim=None):
    """rescale so range [0,1]"""
    if dim is None:
        x = x - torch.min(x)
        m = torch.max(x)
        if m > 1e-8:
            x = x / torch.max(x)
        else:
            x = torch.ones_like(x)
        return x
    else:
        # if single element, don't normalize
        if x.shape[dim] == 1:
            return x

        xmin = torch.min(x, dim=dim)[0]
        x = x - xmin.unsqueeze(dim=dim)
        xmax = torch.max(x, dim=dim)[0]
        xmax = xmax.clamp_min(1e-8)
        x = x / xmax.unsqueeze(dim=dim)
        return x


def normalize_max(x):
    """rescale to range [?,1]"""
    m = torch.max(x)
    if m > 0.0001:
        x = x / m
    else:
        x = torch.ones_like(x)
    return x


def precision_to_variance(x):
    return -torch.log(x + EPSILON) / VP_FACTOR


def variance_to_precision(x):
    return torch.exp(-VP_FACTOR * x)


def gaussian(mean, var, amplitude, x):
    return amplitude * torch.exp(-((x - mean) ** 2) / (2 * (var + EPSILON)))


# def sum_gaussian(means, vars, amplitudes):
# sum_amplitudes = amplitudes.sum(dim=0) + EPSILON
# mean = (means * amplitudes).sum(dim=0) / sum_amplitudes

# result = gaussian(means, vars, amplitudes, mean)
# result = result.sum(dim=0).clamp_(min=0.0, max=1.0)
# return mean, result


def sum_gaussian(means, precisions, dim=0, compuate_precision=True):
    sum_precisions = precisions.sum(dim=dim) + EPSILON

    # precision weighted mean mean
    mean = (means * precisions).sum(dim=dim) / sum_precisions

    # precision weighted mean precision
    # precision = (precisions * precisions).sum(dim=dim) / sum_precisions

    ## alternate method for precision --
    if compuate_precision:
        dist_from_mean = (means - mean.unsqueeze(dim=dim)).abs()
        precision = (1.0 - dist_from_mean.min(dim=dim)[0]) * precisions.max(dim=dim)[0]
    else:
        precision = None

    return mean, precision


def gaussian_kernel(l=5, sig=1.0):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    if l == 1:
        return torch.ones((1, 1))

    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = kernel - kernel.min()
    if np.max(kernel) < 0.001:
        a = 10
    return torch.tensor(kernel / np.max(kernel))


def make_kernel_grid(
    grid_shape: Tuple,
    signal_shape: Tuple,
    scale: float,
    kernel_type: str = KERNEL_RANDOM,
    noise_amplitude: float = 0,
) -> torch.Tensor:
    assert len(grid_shape) == 2
    assert len(signal_shape) == 2
    assert grid_shape[1] == signal_shape[0] * signal_shape[1]

    if kernel_type == KERNEL_RANDOM or kernel_type is None:
        pixels = torch.rand(grid_shape).to(device)  # [0, 1]
        pixels = normalize_scale_base(pixels, scale=scale, base=Config.BASE_ACTIVATION)
    else:
        kernel_pixels = None
        if kernel_type == KERNEL_GAUSSIAN:
            kernel_pixels = gaussian_kernel(
                l=min(signal_shape), sig=min(signal_shape) * 0.3
            )
        elif kernel_type == KERNEL_DOG:
            # positive = gaussian_kernel(l=min(signal_shape), sig=max(min(signal_shape) * 0.3, 2))
            # negative = gaussian_kernel(l=min(signal_shape), sig=1)
            # kernel_pixels = positive - negative
            kernel_pixels = gaussian_kernel(
                l=min(signal_shape), sig=min(signal_shape) * 0.3
            )
            kernel_pixels[int(signal_shape[0] / 2), int(signal_shape[1] / 2)] = 0

        kernel_pixels = normalize(kernel_pixels)
        kernel_pixels = normalize_scale_base(
            kernel_pixels, scale=scale, base=Config.BASE_ACTIVATION
        )

        if signal_shape[0] != signal_shape[1]:
            kernel_pixels_expanded = torch.Tensor(signal_shape)
            top = math.floor((signal_shape[0] - kernel_pixels.shape[0]) / 2)
            left = math.floor((signal_shape[1] - kernel_pixels.shape[1]) / 2)
            kernel_pixels_expanded[
                top : top + kernel_pixels.shape[0], left : left + kernel_pixels.shape[1]
            ] = kernel_pixels
        else:
            kernel_pixels_expanded = kernel_pixels

        kernel_pixels_expanded = kernel_pixels_expanded.view(-1)
        pixels = kernel_pixels_expanded.expand(
            (grid_shape[0], kernel_pixels_expanded.shape[0])
        )

        if noise_amplitude > 0:
            noise_pixels = (
                torch.rand_like(pixels).to(device) * noise_amplitude
                - noise_amplitude * 0.5
            )

            pixels = pixels + noise_pixels

    return pixels


def normalize_scale_base(pixels, scale: float, base: float):
    assert base >= scale * 0.5, "Invalid arguments, resulting range would go below 0"
    assert pixels.min() >= 0.0 and pixels.max() <= 1.0, "Pixels must be [0,1]"

    # pixels                 # [0, 1]
    pixels = pixels * scale  # [0, scale]
    pixels = pixels + base  # [base, scale+base]
    pixels = pixels - scale * 0.5  # [base - scale/2, base + scale/2]

    return pixels


def bounded_contrast_enhancement(x, power=100, dim=None):
    """
    if max value in x is c,
    max value in bounded_contrast_enhancement(x) is also c, but x contrast enhanced

    power denotes how far from linear the exponential curve is
    See https://www.desmos.com/calculator/7ozbz5rnuq
    """

    if dim is None:
        x_max = x.max()
        x = normalize(x)

        power = torch.ones_like(x) * power
        x = (power.pow(x) - 1) / (power - 1)

        x = x * x_max
        return x
    else:
        x_max = x.max(dim=dim)[0]
        x = normalize(x, dim=dim)
        power = torch.ones_like(x) * power
        x = (power.pow(x) - 1) / (power - 1)
        x = x * x_max.unsqueeze(dim=dim)
        return x


def _0213_permute_index(i: int, a: int, b: int, c: int, d: int):
    row = math.floor(i / (c * d))
    col = i % (c * d)

    a1 = math.floor(row / b)
    b1 = row % b
    c1 = math.floor(col / d)
    d1 = col % d

    row_i = a1 * c + c1
    col_i = b1 * d + d1

    index = row_i * b * d + col_i
    return index
