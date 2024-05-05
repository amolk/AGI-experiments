import pytest

from patternmachine.conv_utils import *


def test_get_image_patch_indices_1():
    (image_patch_indices_y, image_patch_indices_x,) = ConvolutionUtils.get_image_patch_indices(
        image_shape=(2, 2), kernel_shape=(1, 1), patch_grid_shape=(2, 2), stride=None
    )
    assert torch.equal(image_patch_indices_y, torch.tensor([[0, 0], [1, 1]]))
    assert torch.equal(image_patch_indices_x, torch.tensor([[0, 1], [0, 1]]))


def test_get_image_patch_indices_2():
    (image_patch_indices_y, image_patch_indices_x,) = ConvolutionUtils.get_image_patch_indices(
        image_shape=(2, 2), kernel_shape=(1, 1), patch_grid_shape=None, stride=(1, 1)
    )
    assert torch.equal(image_patch_indices_y, torch.tensor([[0, 0], [1, 1]]))
    assert torch.equal(image_patch_indices_x, torch.tensor([[0, 1], [0, 1]]))


def test_get_image_patch_indices_3():
    (image_patch_indices_y, image_patch_indices_x,) = ConvolutionUtils.get_image_patch_indices(
        image_shape=(2, 2), kernel_shape=(2, 2), patch_grid_shape=(1, 1), stride=None
    )
    assert torch.equal(image_patch_indices_y, torch.tensor([[0]]))
    assert torch.equal(image_patch_indices_x, torch.tensor([[0]]))


def test_get_image_patch_indices_4():
    (image_patch_indices_y, image_patch_indices_x,) = ConvolutionUtils.get_image_patch_indices(
        image_shape=(2, 2), kernel_shape=(2, 2), patch_grid_shape=None, stride=(1, 1)
    )
    assert torch.equal(image_patch_indices_y, torch.tensor([[0]]))
    assert torch.equal(image_patch_indices_x, torch.tensor([[0]]))


def test_get_image_patch_indices_5():
    (image_patch_indices_y, image_patch_indices_x,) = ConvolutionUtils.get_image_patch_indices(
        image_shape=(4, 5), kernel_shape=(2, 2), patch_grid_shape=None, stride=(2, 2)
    )
    assert torch.equal(image_patch_indices_y, torch.tensor([[0, 0], [2, 2]]))
    assert torch.equal(image_patch_indices_x, torch.tensor([[0, 2], [0, 2]]))


def test_get_patches_info_0():
    image_shape = (4, 5)
    kernel_shape = (2, 2)
    stride = (2, 2)
    (image_patch_indices_y, image_patch_indices_x,) = ConvolutionUtils.get_image_patch_indices(
        image_shape=image_shape, kernel_shape=kernel_shape, patch_grid_shape=None, stride=stride
    )

    patches_info = ConvolutionUtils.get_patches_info(
        image_shape=image_shape,
        kernel_shape=kernel_shape,
        image_patch_indices_y=image_patch_indices_y,
        image_patch_indices_x=image_patch_indices_x,
    )

    assert patches_info.grid_shape == (2, 2)
    assert len(patches_info.patches) == 4

    assert patches_info.patches[0].patch_index == 0
    assert patches_info.patches[0].patch_coordinates == (0, 0)
    assert patches_info.patches[0].patch_coordinates_from == (0, 0)
    assert patches_info.patches[0].patch_coordinates_to == (1, 1)
    assert patches_info.patches[0].image_coordinates_from == (0, 0)
    assert patches_info.patches[0].image_coordinates_to == (1, 1)

    assert patches_info.patches[3].patch_index == 3
    assert patches_info.patches[3].patch_coordinates == (2, 2)
    assert patches_info.patches[3].patch_coordinates_from == (0, 0)
    assert patches_info.patches[3].patch_coordinates_to == (1, 1)
    assert patches_info.patches[3].image_coordinates_from == (2, 2)
    assert patches_info.patches[3].image_coordinates_to == (3, 3)


def test_get_patches_info_1():
    """
    Padded image
    """
    image_shape = (4, 7)
    kernel_shape = (3, 5)
    padding = (1, 2)
    padded_image_shape = (image_shape[0] + padding[0] * 2, image_shape[1] + padding[1] * 2)

    (image_patch_indices_y, image_patch_indices_x,) = ConvolutionUtils.get_image_patch_indices(
        image_shape=padded_image_shape,
        kernel_shape=kernel_shape,
        patch_grid_shape=image_shape,
        stride=None,
    )

    assert image_patch_indices_y.shape == image_shape

    image_patch_indices_y = image_patch_indices_y - padding[0]
    image_patch_indices_x = image_patch_indices_x - padding[1]

    assert torch.equal(
        image_patch_indices_y,
        torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2],
            ]
        ).long(),
    )

    assert torch.equal(
        image_patch_indices_x,
        torch.Tensor(
            [
                [-2, -1, 0, 1, 2, 3, 4],
                [-2, -1, 0, 1, 2, 3, 4],
                [-2, -1, 0, 1, 2, 3, 4],
                [-2, -1, 0, 1, 2, 3, 4],
            ]
        ).long(),
    )

    patches_info = ConvolutionUtils.get_patches_info(
        image_shape=image_shape,
        kernel_shape=kernel_shape,
        image_patch_indices_y=image_patch_indices_y,
        image_patch_indices_x=image_patch_indices_x,
    )

    assert patches_info.grid_shape == image_shape
    assert len(patches_info.patches) == image_shape[0] * image_shape[1]

    # partial top left corner patch
    assert patches_info.patches[0].patch_index == 0
    assert patches_info.patches[0].patch_coordinates == (-1, -2)
    assert patches_info.patches[0].patch_coordinates_from == (1, 2)
    assert patches_info.patches[0].patch_coordinates_to == (2, 4)
    assert patches_info.patches[0].image_coordinates_from == (0, 0)
    assert patches_info.patches[0].image_coordinates_to == (1, 2)

    # first full patch at top left corner
    assert patches_info.patches[9].patch_index == 9
    assert patches_info.patches[9].patch_coordinates == (0, 0)
    assert patches_info.patches[9].patch_coordinates_from == (0, 0)
    assert patches_info.patches[9].patch_coordinates_to == (2, 4)
    assert patches_info.patches[9].image_coordinates_from == (0, 0)
    assert patches_info.patches[9].image_coordinates_to == (2, 4)

    # patch outside the right edge
    assert patches_info.patches[13].patch_index == 13
    assert patches_info.patches[13].patch_coordinates == (0, 4)
    assert patches_info.patches[13].patch_coordinates_from == (0, 0)
    assert patches_info.patches[13].patch_coordinates_to == (2, 2)
    assert patches_info.patches[13].image_coordinates_from == (0, 4)
    assert patches_info.patches[13].image_coordinates_to == (2, 6)

    # partial patch at the bottom right corner
    assert patches_info.patches[27].patch_index == 27
    assert patches_info.patches[27].patch_coordinates == (2, 4)
    assert patches_info.patches[27].patch_coordinates_from == (0, 0)
    assert patches_info.patches[27].patch_coordinates_to == (1, 2)
    assert patches_info.patches[27].image_coordinates_from == (2, 4)
    assert patches_info.patches[27].image_coordinates_to == (3, 6)
