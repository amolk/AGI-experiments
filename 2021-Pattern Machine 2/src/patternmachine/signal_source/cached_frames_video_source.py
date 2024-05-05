from patternmachine.signal_source.cached_signal_source import CachedSignalSource
import torch
import numpy as np
import cv2


class CachedFramesVideoSource(CachedSignalSource):
    def __init__(
        self,
        height=100,
        width=100,
        filepath="/Users/amolk/Downloads/mixkit-tree-branche-under-the-rain-in-the-woods-6782-medium.mp4",
    ) -> None:
        self.height = height
        self.width = width
        self.filepath = filepath
        super().__init__()

    @property
    def item_count(self):
        return 30

    def load_frames(self):
        self.video = self.load_video()

        sequences = [[], [], []]
        for frame in range(0, 50):
            _, image = self.video.read()
            if frame % 5 == 0:
                sequences[0].append(
                    self.make_sgs_from_image(self.crop_video_frame(image, 400, 400).clone())
                )
                sequences[1].append(
                    self.make_sgs_from_image(self.crop_video_frame(image, 200, 800).clone())
                )
                sequences[2].append(
                    self.make_sgs_from_image(self.crop_video_frame(image, 100, 240).clone())
                )

        return sequences[0] + sequences[1] + sequences[2]

    def load_video(self):
        return cv2.VideoCapture(self.filepath)

    def crop_video_frame(self, image, top, left):
        return (
            torch.tensor(image[:, :, 1][top : top + self.height, left : left + self.width]) / 255.0
        )
