# %%
from IPython import get_ipython
from IPython.display import display

print = display

import sys

sys.path.append("/Users/amolk/work/AGI/pattern-machine/src")

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("load_ext", "line_profiler")
get_ipython().run_line_magic("autoreload", "1")
get_ipython().run_line_magic("aimport", "patternmachine.utils")
get_ipython().run_line_magic("aimport", "patternmachine.pattern_similarity")
get_ipython().run_line_magic("matplotlib", "inline")

from patternmachine.similarity_utils import (
    _overlap_similarity,
    overlap_similarity_exp_memoized,
    overlap_similarity_lin_memoized,
    overlap_similarity_lin_interpolate,
    similarity,
)
from patternmachine.utils import make_2d_image, show_image_grid
import torch
import numpy as np
from patternmachine.similarity_utils import MIN_PRECISION, MAX_PRECISION

# %%

resolution = 100
x, x_precision = torch.meshgrid(
    torch.linspace(start=0, end=1, steps=resolution),
    torch.linspace(start=MIN_PRECISION, end=MAX_PRECISION, steps=resolution),
)


test_cases = [
    {"y": 0, "y_precision": MIN_PRECISION},
    {"y": 0.5, "y_precision": MIN_PRECISION},
    {"y": 1, "y_precision": MIN_PRECISION},
    {"y": 0, "y_precision": 0.5},
    {"y": 0.5, "y_precision": 0.5},
    {"y": 1, "y_precision": 0.5},
    {"y": 0, "y_precision": MAX_PRECISION},
    {"y": 0.5, "y_precision": MAX_PRECISION},
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

    _s = _overlap_similarity(x, x_precision, y, y_precision)

    images.append(_s)
    images_vmin.append(0)
    images_vmax.append(1)
    images_title.append("not memoized")

    methods = [
        # overlap_similarity_exp_memoized,
        overlap_similarity_lin_memoized,
        # overlap_similarity_lin_interpolate,
        similarity,
    ]
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
get_ipython().run_line_magic("timeit", "-n 3000 similarity(x, x_precision, y, y_precision)")
get_ipython().run_line_magic(
    "timeit", "-n 3000 _overlap_similarity(x, x_precision, y, y_precision)"
)
get_ipython().run_line_magic(
    "timeit", "-n 3000 overlap_similarity_exp_memoized(x, x_precision, y, y_precision)"
)
get_ipython().run_line_magic(
    "timeit", "-n 3000 overlap_similarity_lin_memoized(x, x_precision, y, y_precision)"
)
# get_ipython().run_line_magic(
#     "timeit", "-n 3000 overlap_similarity_lin_interpolate(x, x_precision, y, y_precision)"
# )
get_ipython().run_line_magic("timeit", "-n 3000 similarity(x, x_precision, y, y_precision)")
# %%
