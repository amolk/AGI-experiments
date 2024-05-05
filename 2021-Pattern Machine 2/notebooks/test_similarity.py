# %%
from IPython import get_ipython
from IPython.display import display

print = display

import sys

sys.path.append("/Users/amolk/work/AGI/pattern-machine/src")

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("load_ext", "line_profiler")
get_ipython().run_line_magic("autoreload", "1")
get_ipython().run_line_magic("aimport", "patternmachine.similarity_utils")
get_ipython().run_line_magic("aimport", "patternmachine.pattern_similarity")
get_ipython().run_line_magic("matplotlib", "inline")

from patternmachine.similarity_utils import MIN_PRECISION, MAX_PRECISION
import torch
from patternmachine.utils import show_image_grid
import torch
import numpy as np
from patternmachine.utils import inverse
from config import Config

# %%
MID_PRECISION = 0.5

# x, x_precision, y, y_precision, sim
test_cases = torch.tensor(
    [
        #
        # 0.0, MIN_PRECISION
        [0.0, MIN_PRECISION, 0.0, MIN_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MIN_PRECISION, 0.0, MAX_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MIN_PRECISION, 1.0, MIN_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MIN_PRECISION, 1.0, MAX_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MIN_PRECISION, 0.0, MID_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MIN_PRECISION, 0.5, MID_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MIN_PRECISION, 1.0, MID_PRECISION, 0.99],  # MIN_PRECISION present
        #
        # 0.0, MAX_PRECISION
        [0.0, MAX_PRECISION, 0.0, MIN_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MAX_PRECISION, 0.0, MAX_PRECISION, 0.98],  # same value
        [0.0, MAX_PRECISION, 1.0, MIN_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MAX_PRECISION, 1.0, MAX_PRECISION, 0.0],  # MAX_PRECISION, opposite value
        [0.0, MAX_PRECISION, 0.0, MID_PRECISION, 1.0],  # same value
        [0.0, MAX_PRECISION, 0.5, MID_PRECISION, 0.56],  # MAX_PRECISION, values mismatch
        [0.0, MAX_PRECISION, 1.0, MID_PRECISION, 0.25],  # MAX_PRECISION, values mismatch
        #
        # 0.0, MID_PRECISION
        [0.0, MID_PRECISION, 0.0, MIN_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MID_PRECISION, 0.0, MAX_PRECISION, 1.0],  # same value
        [0.0, MID_PRECISION, 1.0, MIN_PRECISION, 0.99],  # MIN_PRECISION present
        [0.0, MID_PRECISION, 1.0, MAX_PRECISION, 0.25],  # MAX_PRECISION present, value mismatch
        [0.0, MID_PRECISION, 0.0, MID_PRECISION, 0.56],  # same value but MID_PRECISION
        [0.0, MID_PRECISION, 0.5, MID_PRECISION, 0.40],  # MID_PRECISION, values apart
        [0.0, MID_PRECISION, 1.0, MID_PRECISION, 0.25],  # MID_PRECISION, values further apart
    ]
)


def similarity(x, x_precision, y, y_precision):
    min_precision = torch.stack([x_precision, y_precision]).min(axis=0)[0]
    max_precision = torch.stack([x_precision, y_precision]).max(axis=0)[0]

    err = (x - y).abs()
    sim = (1 - min_precision) * 0.2 + ((1 - err) * (min_precision + max_precision) * 0.5).pow(2)
    # sim = sim.pow(2)
    sim.clamp_(min=0, max=1)

    # shape = x.shape
    # x_abs = ((x - Config.BASE_ACTIVATION).abs() / (1 - Config.BASE_ACTIVATION)).view(shape)
    # y_abs = ((y - Config.BASE_ACTIVATION).abs() / (1 - Config.BASE_ACTIVATION)).view(shape)
    # sim2 = sim * x_abs * y_abs  # / (x_abs.max(dim=-1)[0] * y_abs.max(dim=-1)[0]).unsqueeze(-1)

    # a = 1.0
    # sim = a * sim2 + (1 - a) * sim
    return sim


def similarity2(x, x_precision, y, y_precision):
    min_precision = torch.stack([x_precision, y_precision]).min(axis=0)[0]
    max_precision = torch.stack([x_precision, y_precision]).max(axis=0)[0]
    avg_precision = x_precision * y_precision

    s = 7
    max_inverse = inverse(max_precision, scale=s)
    min_inverse = inverse(min_precision, scale=s)
    err = (x - y).abs()

    # sim = (1 - min_precision) + min_precision * (1 - err) * max_precision  # (1 - max_inverse)
    sim = min_inverse + (1 - min_inverse) * (1 - err) * (1 - max_inverse)  # (1 - max_inverse)

    return sim  # .pow(2)  # .clamp(min=0, max=1)


def similarity_squ(x, x_precision, y, y_precision):
    return similarity(x, x_precision, y, y_precision).pow(2)


def similarity2_squ(x, x_precision, y, y_precision):
    return similarity2(x, x_precision, y, y_precision).pow(2)


def test_similarity():
    error = 0
    pass_count = 0
    for test_case in test_cases:
        sim = similarity(
            x=test_case[0],
            x_precision=test_case[1],
            y=test_case[2],
            y_precision=test_case[3],
        )

        if (sim - test_case[4]).abs() < 0.01:
            pass_count += 1
            print(f"Test {test_case} passed. Expected {test_case[4]}, Got {sim}")
        else:
            print(f"*** Test {test_case} failed. Expected {test_case[4]}, Got {sim}")

        error += (sim - test_case[4]).abs()

    print(f"{pass_count}/{len(test_cases)} correct")
    print(f"Error: {error}")


test_similarity()


resolution = 100
x, x_precision = torch.meshgrid(
    torch.linspace(start=0, end=1, steps=resolution),
    torch.linspace(start=MIN_PRECISION, end=MAX_PRECISION, steps=resolution),
)


test_cases = [
    {"y": 0, "y_precision": MIN_PRECISION},
    {"y": 0, "y_precision": 0.5},
    {"y": 0, "y_precision": MAX_PRECISION},
    #
    {"y": 0.5, "y_precision": MIN_PRECISION},
    {"y": 0.5, "y_precision": 0.05},
    {"y": 0.5, "y_precision": 0.15},
    {"y": 0.5, "y_precision": 0.25},
    {"y": 0.5, "y_precision": 0.5},
    {"y": 0.5, "y_precision": 0.75},
    {"y": 0.5, "y_precision": 0.85},
    {"y": 0.5, "y_precision": 0.95},
    {"y": 0.5, "y_precision": MAX_PRECISION},
    #
    {"y": 1, "y_precision": MIN_PRECISION},
    {"y": 1, "y_precision": 0.5},
    {"y": 1, "y_precision": MAX_PRECISION},
]
for test_case in test_cases:
    images = []
    images_vmin = []
    images_vmax = []
    images_title = []

    print(test_case)

    y = torch.ones_like(x) * test_case["y"]
    y_precision = torch.ones_like(x_precision) * test_case["y_precision"]

    _s = similarity(x, x_precision, y, y_precision)

    images.append(_s)
    images_vmin.append(0)
    images_vmax.append(1)
    images_title.append("not memoized")

    methods = [similarity_squ, similarity2, similarity2_squ]
    for method in methods:
        s = method(x, x_precision, y, y_precision)

        images.append(s)
        images_vmin.append(0)
        images_vmax.append(1)
        images_title.append(method.__name__)

        images.append((_s - s).abs())
        images_vmin.append(0)
        images_vmax.append(0.2)
        images_title.append(
            f"error min {(_s-s).abs().min():.2f}, max {(_s-s).abs().max():.2f}, mean {(_s-s).abs().mean():.2f}"
        )

    print(images_title)
    show_image_grid(
        images=np.stack(images),
        vmin=images_vmin,
        vmax=images_vmax,
        grid_width=1 + len(methods) * 2,
        grid_height=1,
    )


# %%
def featurize(x):
    # return x
    # return torch.hstack([x, (-x).exp()])
    return torch.hstack([x, x.exp(), (-x).exp(), x.pow(2)])


import numpy as np
from sklearn.linear_model import LinearRegression

X = featurize(test_cases[:, 0:4])
# y = 1 * x_0 + 2 * x_1 + 3
y = test_cases[:, 4]
reg = LinearRegression(positive=True).fit(X, y)
reg.score(X, y)
# %%
for test_case in test_cases:
    print(test_case)
    pred = reg.predict(featurize(test_case[0:4].unsqueeze(dim=0)))
    print(f"{test_case[4]} -- {pred[0]}")
    print("---")
# %%
reg.score(X, y)  # %%

# %%
reg.coef_
# %%
import torch
import torch.nn as nn


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = torch.nn.Linear(16, 40)
        self.linear2 = torch.nn.Linear(40, 1)
        # self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        y_pred = torch.sigmoid(self.linear2(x))
        # y_pred = self.linear(x)
        return y_pred


model = LinearRegressionModel()
learning_rate = 0.0001
l = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000
for epoch in range(num_epochs):
    # forward feed
    y_pred = model(X.requires_grad_())

    # calculate the loss
    loss = l(y_pred, y.unsqueeze(dim=-1))

    # backward propagation: calculate gradients
    loss.backward()

    # update the weights
    optimizer.step()

    # clear out the gradients from the last step loss.backward()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print("epoch {}, loss {}".format(epoch, loss.item()))
        print(torch.hstack([model(X), y.unsqueeze(dim=-1)]))
# %%
(X, y)
# %%
# %%
