import math
import torch
import numpy as np
import cv2

from patternmachine.signal_source.signal_source import SignalSource


class RotatingRectangleSignalSource(SignalSource):
    def __init__(self, height=100, width=100, start_angle=0, end_angle=180, angle_step=10) -> None:
        assert height == width
        self.height = height
        self.width = width
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.angle_step = angle_step
        self.angle = start_angle

        super().__init__()

    def create_rotated_rect_image(self, size, angle, color=1, thickness=-1):
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

    @property
    def item_count(self):
        return abs(math.floor((self.end_angle - self.start_angle) / self.angle_step)) * 2

    def item(self):
        while True:
            sgs = self.make_sgs_from_image(
                self.create_rotated_rect_image(size=self.height, angle=self.angle)
            )

            self.angle += self.angle_step
            if self.angle > self.end_angle or self.angle < self.start_angle:
                self.angle_step *= -1
                self.angle += 2 * self.angle_step

            yield sgs

    def seek(self, frame_index=0):
        self.angle = self.start_angle + frame_index * self.angle_step
