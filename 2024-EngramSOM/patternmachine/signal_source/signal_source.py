import math
from typing import List

import numpy as np
import torch

from patternmachine.utils import show_image_grid


class SignalSource:
    @property
    def item_count(self):
        raise NotImplementedError()

    def item(self):
        raise NotImplementedError()

    def imshow(self, frame_count=None, filename=None):
        signal_generator = self.item()

        images = []
        if frame_count is None:
            frame_count = self.item_count
        sqrt = math.ceil(math.sqrt(frame_count))

        for _ in range(sqrt * sqrt):
            image = next(signal_generator)
            images.append(image)

        show_image_grid(
            torch.stack(images),
            vmin=0,
            vmax=1,
            grid_width=sqrt,
            grid_height=sqrt,
            filename=filename,
        )
