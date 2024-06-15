import math

import cv2
import torch

from patternmachine.signal_source.signal_source import SignalSource


class VideoSource(SignalSource):
    def __init__(
        self,
        height=100,
        width=100,
        filepath="/Users/amolk/Downloads/mixkit-tree-branche-under-the-rain-in-the-woods-6782-medium.mp4",
        stride=4,
        invert=False,
    ) -> None:
        self.filepath = filepath
        self.height = height
        self.width = width
        self.video = self.load_video()
        self.stride = stride
        self.invert = invert

        self.seek(0)
        image = next(self.item())
        self.height, self.width = image.shape

    @property
    def item_count(self):
        return math.floor(self.video.get(cv2.CAP_PROP_FRAME_COUNT) / self.stride)

    def seek(self, frame_index):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def item(self):
        while True:
            for _ in range(self.stride - 1):
                self.video.read()
            _, image = self.video.read()
            if image is None:
                self.seek(0)
                _, image = self.video.read()
            if self.invert:
                image = cv2.bitwise_not(image)
            yield self.crop_video_frame(image, 0, 70)

    def load_video(self):
        return cv2.VideoCapture(self.filepath)

    def crop_video_frame(self, image, top, left):
        return (
            torch.tensor(
                image[:, :, 1][top : top + self.height, left : left + self.width]
            )
            / 255.0
        )
