import pdb

import numpy as np
import torch

from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.similarity_utils import similarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PatternSimilarityHP:
    def __init__(self, enable_precision_weighted_distance=True):
        self.enable_precision_weighted_distance = enable_precision_weighted_distance
        self.component_weights = {"__output__": 0.0}


class PatternSimilarity:
    def __init__(
        self,
        signal: SignalGridSet,
        patterns: SignalGridSet,
        hp: PatternSimilarityHP = None,
    ):
        """
        signal must be        grid_shape=(grid_shape)                               signal_shape=(composite_signal_shapes)
        patterns must be      grid_shape=(grid_shape, per_item_pattern_grid_shape)  signal_shape=(composite_signal_shapes)

        Each signal is compared with corresponding per_x_pattern_grid_shape patterns.
        Each comparison is done across composite signal components
        """
        # print("signal", signal)
        # print(
        #     "signal.components['__output__']", signal.components["__output__"]
        # ) if "__output__" in signal.components else None
        # print("patterns", patterns)
        # print(
        #     "patterns.components['__output__']", patterns.components["__output__"]
        # ) if "__output__" in signal.components else None
        if hp:
            self.hp = hp
        else:
            self.hp = PatternSimilarityHP()

        assert (
            signal.component_count == patterns.component_count
        ), f"signal.components {signal.components.keys()} != patterns.components {patterns.components.keys()}"

        for name in signal.signal_shape:
            assert np.prod(signal.signal_shape[name]) == np.prod(
                patterns.signal_shape[name]
            ), f"signal.signal_shape {signal.signal_shape} != patterns.signal_shape {patterns.signal_shape}"

        assert (
            patterns.grid_shape[0 : len(signal.grid_shape)] == signal.grid_shape
        ), f"patterns.grid_shape {patterns.grid_shape} must match 0-n dimensions with signal.grid_shape {signal.grid_shape}"
        per_item_pattern_grid_shape = patterns.grid_shape[len(signal.grid_shape) :]
        # print("per_item_pattern_grid_shape", per_item_pattern_grid_shape)
        per_item_pattern_grid_size = np.prod(per_item_pattern_grid_shape)

        self.dist_1 = {}
        self.dist_d = {}
        self.dist = []
        self.sim_components = {}
        self.precision_components = {}
        self.component_weights = self.hp.component_weights.copy()

        # find similarity based on each signal component
        for name, x_component in signal.components.items():
            # default component weight is 1.0
            if name not in self.component_weights:
                self.component_weights[name] = 1.0

            # print("Component ", name)
            # print("================")
            y_component = patterns.components[name]
            # print("x_component", x_component)
            # print("y_component", y_component)

            xs = x_component.pixels.shape
            # print("xs", xs)
            x_component_pixels = x_component.pixels
            if len(xs) == 1:
                xs = (1, xs[0])
                x_component_pixels = x_component.pixels.view(xs)

            x_component_pixels_expanded = (
                x_component_pixels.unsqueeze(dim=1)
                .expand((xs[0], per_item_pattern_grid_size, xs[1]))
                .reshape(-1, xs[1])
            )  # expensive to reshape. How to vectorize better?

            assert x_component_pixels_expanded.shape == y_component.pixels.shape

            x_component_precision_expanded = None
            y_component_precision = None

            x_component_precision = x_component.precision
            if x_component_precision is not None:
                # scale precision by contrast
                # x_component_precision = x_component_precision * x_component.pixel_contrast.mean(
                #     dim=0
                # )

                x_component_precision_expanded = (
                    x_component_precision.unsqueeze(dim=1)
                    .expand((xs[0], per_item_pattern_grid_size, xs[1]))
                    .reshape(-1, xs[1])
                )  # expensive to reshape. How to vectorize better?

            y_component_precision = y_component.precision
            if x_component_precision_expanded is not None and y_component_precision is not None:
                assert x_component_precision_expanded.shape == y_component_precision.shape

            # print("y_component.pixels", y_component.pixels.shape)
            sim_1 = similarity(
                x=x_component_pixels_expanded,
                x_precision=x_component_precision_expanded,
                y=y_component.pixels,
                y_precision=y_component_precision,
            )

            sim_component = sim_1.mean(dim=-1)
            # sim_component = torch.exp(-dist_component) * precision_component
            # print("sim_component", sim_component)

            # self.dist_1[name] = dist_1_component
            # self.dist_d[name] = dist_d_component
            # self.dist.append(dist_component)
            self.sim_components[name] = sim_component
            # self.precision_components[name] = precision_component

        # print("self.sim_components", self.sim_components)
        # final similarity is mean of signal component similarities
        # this equalizes class weights for all components (e.g. modalities)
        weighted_sim_components = [
            self.sim_components[name] * self.component_weights[name] for name in signal.components
        ]
        self.sim = torch.stack(weighted_sim_components).sum(dim=0) / torch.sum(
            torch.tensor(list(self.component_weights.values()))
        )

        # self.sim = self.sim * patterns.alpha
        # self.precision = torch.stack(list(self.precision_components.values())).mean(dim=0)
        # print("self.sim", self.sim)

        # $HACK contrast enhancement
        # self.sim = self.sim - self.sim.min(dim=0).values + 0.01

    def l2_cross_distance(self, x, x_precision, y, y_precision):
        xs = x.shape
        assert len(xs) == 2
        assert (x_precision is None) or (
            x_precision.shape == xs
        ), "Precision, if specified, must be same shape as patterns"

        ys = y.shape
        assert len(ys) == 2
        assert (y_precision is None) or (
            y_precision.shape == ys
        ), "Precision, if specified, must be same shape as patterns"

        assert xs[1] == ys[1], "Patch size, i.e. dim 1, must match"

        n = xs[0]
        m = ys[0]
        d = xs[1]

        x = x.unsqueeze(1).expand(n, m, d)
        x_precision = x_precision.unsqueeze(1).expand(n, m, d)

        y = y.unsqueeze(0).expand(n, m, d)
        y_precision = y_precision.unsqueeze(0).expand(n, m, d)

        return self.l2_distance(x, x_precision, y, y_precision)

    def l2_distance(self, x, x_precision, y, y_precision):
        # print("x", x)
        # print("x_precision", x_precision)
        # print("y", y)
        # print("y_precision", y_precision)
        dist_1 = (x - y).abs()
        dist_d = torch.pow(dist_1, 2)
        # print("dist_d", dist_d)

        precision_d = None
        if self.hp.enable_precision_weighted_distance:
            if x_precision is not None:
                dist_d = dist_d * x_precision
                precision_d = x_precision

            if y_precision is not None:
                dist_d = dist_d * y_precision
                if precision_d is None:
                    precision_d = y_precision
                else:
                    precision_d = precision_d * y_precision

        precision = None
        if precision_d is not None:
            precision = precision_d.mean(-1)

        dist = dist_d.sum(-1).sqrt()
        # print("dist_d after", dist_d)
        return dist_1, dist_d, dist, precision_d, precision

    def __repr__(self):
        s = [f"self.dist_1: {self.dist_1}"]
        return "".join(s)
