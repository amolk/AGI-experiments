import math
import torch
import numpy as np
import cv2
import random

from patternmachine.signal_source.signal_source import SignalSource


class MovingRectangleSignalSource(SignalSource):
    def __init__(self, height=100, width=100, step=10) -> None:
        assert height == width
        self.height = height
        self.width = width
        self.angle = 45
        self.x = -width
        self.step = step
        super().__init__()

    def create_rect_image(self, size, x, angle, color=1, thickness=-1):
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
        move_mat = np.float32([[1, 0, x], [0, 1, x]])
        img = cv2.warpAffine(img, move_mat, img.shape)
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

    @property
    def item_count(self):
        return 1000

    def item(self):
        while True:
            sgs = self.make_sgs_from_image(
                self.create_rect_image(size=self.height, angle=self.angle, x=self.x)
            )

            self.x += self.step
            if self.x > self.width * 3 / 2:
                self.x = -self.width / 2
                self.angle = random.randint(0, 360)

            yield sgs

    def seek(self, frame_index=0):
        self.angle = self.start_angle + frame_index * self.angle_step
