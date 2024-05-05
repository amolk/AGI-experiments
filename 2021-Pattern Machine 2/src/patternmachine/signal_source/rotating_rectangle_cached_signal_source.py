from typing import List
import cv2
import torch
import numpy as np

from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.signal_source.cached_signal_source import CachedSignalSource
from config import Config


class RotatingRectangleCachedSignalSource(CachedSignalSource):
    def __init__(self, height=100, width=100, start_angle=0, end_angle=380, angle_step=10) -> None:
        self.height = height
        self.width = width
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.angle_step = angle_step

        super().__init__()

    def load_frames(self):
        frames: List[SignalGridSet] = []
        image_shape = (self.height, self.width)
        sgs = self.make_sgs_from_image(
            self.create_rotated_rect_image(
                size=image_shape[0], angle=self.start_angle - self.angle_step
            )
        )
        for angle in range(self.start_angle, self.end_angle, self.angle_step):
            next_sgs = self.make_sgs_from_image(
                self.create_rotated_rect_image(size=image_shape[0], angle=angle)
            )

            frames.append(next_sgs)
            # next_sgs = self.make_sgs_from_image(create_line_image(size=mu_shape[0], angle=angle))
            # next_sgs = next_sgs * (1 - Config.BASE_ACTIVATION) + Config.BASE_ACTIVATION
            # sgs.trace_(next_sgs)

            # frames.append(sgs.clone())

        # for angle in range(
        #     self.end_angle - self.angle_step, self.start_angle + self.angle_step, -self.angle_step
        # ):
        #     next_sgs = self.make_sgs_from_image(
        #         self.create_rotated_rect_image(size=image_shape[0], angle=angle)
        #     )

        #     frames.append(next_sgs)

        # frames = frames[int(len(frames) / 2) :]
        return frames

    def create_rotated_rect_image(self, size, angle, color=1.0, thickness=-1):
        original_size = size
        size = size * 2
        img = np.zeros((size, size))
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        cv2.rectangle(
            img,
            pt1=(0, 0),
            pt2=(size - 1, int(image_center[1])),
            color=1.0,
            thickness=thickness,
        )
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        img = img[
            int(original_size / 2) + 1 : int(original_size / 2) + original_size + 1,
            int(original_size / 2) + 1 : int(original_size / 2) + original_size + 1,
        ]

        # [0,1] => [0.3, color]
        img = img * (color - Config.BASE_ACTIVATION) + Config.BASE_ACTIVATION

        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        return torch.tensor(img).float()
