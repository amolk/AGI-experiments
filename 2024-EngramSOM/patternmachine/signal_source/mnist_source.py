import math

import cv2
import numpy as np
import torch
from datasets import load_dataset

from patternmachine.signal_source.signal_source import SignalSource


class MNISTSource(SignalSource):
    def __init__(
        self,
        invert=False,
    ) -> None:
        self.height = 28
        self.width = 28
        self.invert = invert
        dataset = load_dataset("mnist")
        self.train_images = np.asarray(dataset["train"]["image"]) / 255.0
        self.train_labels = np.asarray(dataset["train"]["label"])
        self.test_images = np.asarray(dataset["test"]["image"]) / 255.0
        self.test_labels = np.asarray(dataset["test"]["label"])

    @property
    def item_count(self):
        return self.train_images.shape[0]

    def item(self):
        while True:
            for i in range(self.train_images.shape[0]):
                image = self.train_images[i]
                if self.invert:
                    image = 1 - image
                yield torch.tensor(image, dtype=torch.float32)
