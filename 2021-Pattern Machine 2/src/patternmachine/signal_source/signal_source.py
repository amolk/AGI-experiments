import math
from typing import List
import torch
import numpy as np
from patternmachine.signal_grid import SignalGridHP, SignalGrid
from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.similarity_utils import MAX_PRECISION
from patternmachine.utils import show_image_grid


class SignalSource:
    @property
    def item_count(self):
        raise NotImplementedError()

    def item(self):
        raise NotImplementedError()

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

    def imshow(self, frame_count=None):
        signal_generator = self.item()

        images = []
        if frame_count is None:
            frame_count = self.item_count
        sqrt = math.ceil(math.sqrt(frame_count))

        for _ in range(sqrt * sqrt):
            image = next(signal_generator)
            images.append(image.components["mu"].pixels_as_image)

        show_image_grid(torch.stack(images), vmin=0, vmax=1, grid_width=sqrt, grid_height=sqrt)
