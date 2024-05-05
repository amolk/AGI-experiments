from typing import Tuple

import matplotlib.pyplot as plt
import torch
# from tqdm.notebook import tqdm
from tqdm import tqdm

from patternmachine.layer import Layer, LayerHP
from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.signal_utils import SignalUtils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SingleLayerRegression:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, pattern_grid_shape: Tuple = (1, 10)):
        assert X.shape[0] == Y.shape[0], "Batch size must be same for X and Y"
        assert len(X.shape) >= 1, "X must be at least 1D batch of scalars"
        assert len(Y.shape) >= 1, "Y must be at least 1D batch of scalars"
        self.X = X
        self.Y = Y
        self.pattern_grid_shape = pattern_grid_shape

        self.batch_size = X.shape[0]

        # X
        self.X_signal_shape = X.shape[1:]
        if len(self.X_signal_shape) == 0:
            self.X_signal_shape = (1, 1)
        elif len(self.X_signal_shape) == 1:
            self.X_signal_shape = [1] + list(self.X_signal_shape)
        self.X_signal_shape = tuple(self.X_signal_shape)

        # Y
        self.Y_signal_shape = Y.shape[1:]
        if len(self.Y_signal_shape) == 0:
            self.Y_signal_shape = (1, 1)
        elif len(self.Y_signal_shape) == 1:
            self.Y_signal_shape = [1] + list(self.Y_signal_shape)
        self.Y_signal_shape = tuple(self.Y_signal_shape)

        # Layer
        self.layer_hp = self.create_layer_hp(per_patch_pattern_grid_shape=pattern_grid_shape)
        self.layer = Layer(hp=self.layer_hp)

    def create_layer_hp(
        self,
        input_coverage_factor=1.0,
        patch_grid_shape=(1, 1),
        per_patch_pattern_grid_shape=(1, 10),
        output_patch_neighborhood_shape=(1, 1),
        output_tau=1.0,
    ):
        hp = LayerHP(
            input_signal_shapes={"x": self.X_signal_shape, "y": self.Y_signal_shape},
            input_coverage_factor=input_coverage_factor,
            patch_grid_shape=patch_grid_shape,
            per_patch_pattern_grid_shape=per_patch_pattern_grid_shape,
            output_patch_neighborhood_shape=output_patch_neighborhood_shape,
            output_decay=output_tau,
        )  # set output_decay=1.0 for IID data
        return hp

    def epoch(self):
        error = 0
        prec = 0

        for i in tqdm(range(self.batch_size)):
            input = SignalGridSet.from_pixels_list(
                {
                    "x": self.X[i].view(self.X_signal_shape),
                    "y": self.Y[i].view(self.Y_signal_shape),
                }
            )

            self.layer.forward(input)

            predictions = []
            sample_count = 10
            winner_indices = torch.multinomial(
                self.layer.activation_begin, sample_count, replacement=True
            )
            for winner_index in winner_indices:
                # print("winner_index", winner_index)
                top_down_prediction = self.layer.patterns.pixels_begin[winner_index]
                # print("prediction", top_down_prediction.components[1].pixels)
                predictions.append(top_down_prediction.components["y"].pixels)

            expected = (self.X[i] + self.Y[i]).item() / 2.0
            predictions = torch.stack(predictions)
            variance = torch.var(predictions)
            precision = SignalUtils.compute_precision(variance * 100)
            prediction = predictions.mean(dim=0)
            error = error + (prediction.item() - expected) ** 2
            prec = prec + precision

            if i < 10:
                print("Sample --")
                print("x, y", input.pixels["x"].item(), input.pixels["y"].item())
                print("Expected prediction", expected)
                print(f"prediction ({sample_count} samples)", prediction.item())
                print("prediction precision", precision.item())

                # print("layer output", self.layer.output.pixels.view(self.layer.hp.per_patch_pattern_grid_shape))
                # plt.imshow(
                #     self.layer.output.pixels.view(self.layer.hp.per_patch_pattern_grid_shape),
                #     vmin=0,
                #     vmax=1,
                #     cmap=plt.cm.viridis,
                # )
                # plt.axis("off")
                # plt.show()

            # print("patterns", self.layer.patterns.pixels)
        rms_error = (error / self.batch_size) ** 0.5
        mean_precision = prec / self.batch_size

        print("RMS Error", rms_error)
        print("Mean precision", mean_precision)

        return rms_error, mean_precision
