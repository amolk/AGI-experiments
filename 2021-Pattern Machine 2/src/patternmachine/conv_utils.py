import math
import pdb
from typing import NamedTuple, Tuple
from collections import namedtuple

import torch

from skimage.util.shape import view_as_windows

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImagePatchCoordinates(
    namedtuple(
        "ImagePatchCoordinates",
        [
            "patch_coordinates",
            "patch_index",
            "patch_coordinates_from",
            "patch_coordinates_to",
            "image_coordinates_from",
            "image_coordinates_to",
        ],
    )
):
    def __repr__(self):
        a = []
        a.append("ImagePatchCoordinates")
        a.append(f"    patch_coordinates {self.patch_coordinates}")
        a.append(f"    patch_index       {self.patch_index}")
        a.append(
            f"    patch area from   {self.patch_coordinates_from} to {self.patch_coordinates_to}"
        )
        a.append(
            f"    image area from   {self.image_coordinates_from} to {self.image_coordinates_to}"
        )

        return "\n".join(a)


class ImagePatchesInfo(
    namedtuple("ImagePatchesInfo", ["image_shape", "kernel_shape", "grid_shape", "patches"])
):
    def __repr__(self):
        a = []
        a.append("ImagePatchesInfo")
        a.append(f"    image shape  {self.image_shape}")
        a.append(f"    kernel shape {self.kernel_shape}")
        a.append(f"    grid shape   {self.grid_shape}")
        a.append(f"    patches:     {self.patches}")

        return "\n".join(a)


class ConvolutionUtils:
    @staticmethod
    def conv_slice(image, kernel_shape, padding=None, patch_grid_shape=None, stride=None):
        assert (
            patch_grid_shape is None or stride is None
        ), "Specify either patch grid shape, i.e. number of elements, or stride, i.e. distance between elements"

        if padding is not None:
            assert (
                kernel_shape[0] % 2 == 1 and kernel_shape[1] % 2 == 1
            ), "Kernel shape must be odd"
            assert (
                padding[0] == (kernel_shape[0] - 1) / 2 and padding[1] == (kernel_shape[1] - 1) / 2
            ), f"We support padding only for neighborhood patches, not arbitrary values: padding {padding}, kernel shape {kernel_shape}"
            assert stride is None, "stride is 1 for neighborhood patches"

            # assume padded image
            padded_image_shape = (image.shape[0] + 2 * padding[0], image.shape[1] + 2 * padding[1])
            (
                image_patch_indices_y,
                image_patch_indices_x,
            ) = ConvolutionUtils.get_image_patch_indices(
                image_shape=padded_image_shape,
                kernel_shape=kernel_shape,
                patch_grid_shape=patch_grid_shape,
                stride=stride,
            )

            # adjust patch coordinates back to unpadded image coordinates
            image_patch_indices_y = image_patch_indices_y - padding[0]
            image_patch_indices_x = image_patch_indices_x - padding[1]
        else:
            (
                image_patch_indices_y,
                image_patch_indices_x,
            ) = ConvolutionUtils.get_image_patch_indices(
                image_shape=image.shape,
                kernel_shape=kernel_shape,
                patch_grid_shape=patch_grid_shape,
                stride=stride,
            )

        patches_info = ConvolutionUtils.get_patches_info(
            image_shape=image.shape,
            kernel_shape=kernel_shape,
            image_patch_indices_y=image_patch_indices_y,
            image_patch_indices_x=image_patch_indices_x,
        )

        image_patches = ConvolutionUtils.get_image_patches(image=image, patches_info=patches_info)

        return image_patches, patches_info

    @staticmethod
    def get_image_patch_indices(image_shape, kernel_shape, patch_grid_shape, stride):
        assert len(image_shape) == 2, "Must be (image height, image width)"
        assert len(kernel_shape) == 2, "Only 2D kernels allowed"
        assert (
            patch_grid_shape is None or stride is None
        ), "Specify either patch grid shape, i.e. number of elements, or stride, i.e. distance between elements"

        image_height, image_width = image_shape
        kernel_height, kernel_width = kernel_shape

        if patch_grid_shape is not None:
            assert len(patch_grid_shape) == 2, "Only 2D patch grids allowed"
            patch_grid_height, patch_grid_width = patch_grid_shape

            # We will find patch top-left coordinates
            # First patch has top-left coordinates (0, 0)
            # Last patch has top-left coordinates (image_height - kernel_height)
            # (h, w)th patch has top-left coordinates
            #      (h / patch_grid_height) * (image_height - kernel_height) ,
            #      (w / patch_grid_width)  * (image_width  - kernel_width)

            indices_y, indices_x = torch.meshgrid(
                torch.linspace(0, image_height - kernel_height, patch_grid_height).long(),
                torch.linspace(0, image_width - kernel_width, patch_grid_width).long(),
            )
        else:
            assert len(stride) == 2, "Need 2D stride"
            indices_y, indices_x = torch.meshgrid(
                torch.arange(0, image_height - kernel_height + 0.01, stride[0]).long(),
                torch.arange(0, image_width - kernel_width + 0.01, stride[1]).long(),
            )

        return indices_y, indices_x

    @staticmethod
    def get_patches_info(image_shape, kernel_shape, image_patch_indices_y, image_patch_indices_x):
        (ih, iw) = image_shape

        patch_coordinates = []

        for patch_index, (y, x) in enumerate(
            zip(
                image_patch_indices_y.reshape(-1).tolist(),
                image_patch_indices_x.reshape(-1).tolist(),
            )
        ):
            patch_coordinates_from = tuple(ConvolutionUtils.rectified_linear([-y, -x]))
            relu1 = ConvolutionUtils.rectified_linear(
                [y + kernel_shape[0] - ih, x + kernel_shape[1] - iw]
            )
            patch_coordinates_to = tuple(
                [kernel_shape[0] - 1 - relu1[0], kernel_shape[1] - 1 - relu1[1]]
            )

            image_coordinates_from = tuple(
                [
                    y + patch_coordinates_from[0],
                    x + patch_coordinates_from[1],
                ]
            )
            image_coordinates_to = tuple(
                [
                    y + patch_coordinates_to[0],
                    x + patch_coordinates_to[1],
                ]
            )

            patch_coordinates.append(
                ImagePatchCoordinates(
                    patch_coordinates=(y, x),
                    patch_index=patch_index,
                    patch_coordinates_from=patch_coordinates_from,
                    patch_coordinates_to=patch_coordinates_to,
                    image_coordinates_from=image_coordinates_from,
                    image_coordinates_to=image_coordinates_to,
                )
            )

        ipi = ImagePatchesInfo(
            image_shape=image_shape,
            kernel_shape=kernel_shape,
            grid_shape=image_patch_indices_y.shape,
            patches=patch_coordinates,
        )
        # print(ipi)
        return ipi

    @staticmethod
    def get_image_patches(image, patches_info):
        kernel_shape = patches_info.kernel_shape
        patches = torch.zeros((len(patches_info.patches),) + kernel_shape)
        # print("patches.shape", patches.shape)
        for p in patches_info.patches:
            patches[p.patch_index][
                p.patch_coordinates_from[0] : p.patch_coordinates_to[0] + 1,
                p.patch_coordinates_from[1] : p.patch_coordinates_to[1] + 1,
            ] = image[
                p.image_coordinates_from[0] : p.image_coordinates_to[0] + 1,
                p.image_coordinates_from[1] : p.image_coordinates_to[1] + 1,
            ]

        return patches.view(patches.shape[0], -1).to(device)

    @staticmethod
    def rectified_linear(t):
        result = []
        for i in t:
            if i < 0:
                result.append(0)
            else:
                result.append(i)
        return result