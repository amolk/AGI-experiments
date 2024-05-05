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

import torch
import torch
import numpy as np
from config import Config
from matplotlib import pyplot as plt
from patternmachine.signal_grid_set import SignalGridSet
from patternmachine.signal_source.rotating_rectangle_cached_signal_source import (
    RotatingRectangleCachedSignalSource,
)
from patternmachine.signal_source.moving_rectangle_signal_source import MovingRectangleSignalSource
from patternmachine.similarity_utils import MIN_PRECISION, MAX_PRECISION
from patternmachine.utils import inverse
from patternmachine.utils import show_image_grid

# %%
image_shape = [50, 50]
# signal_source = RotatingRectangleCachedSignalSource(
#     height=image_shape[0], width=image_shape[1], angle_step=7
# )
signal_source = MovingRectangleSignalSource(height=image_shape[0], width=image_shape[1], step=1)

signal_source_items = signal_source.item()

traced_signal = next(signal_source_items).clone()
traced_signal.components["mu"].pixels.epsilon = 0.5
previous_traced_signal = traced_signal.clone()

td_signal = next(signal_source_items).clone()
td_signal.components["mu"].pixels.data *= 0

td_signal_trace = td_signal.clone()
td_signal_trace.components["mu"].pixels.epsilon = 0.5

for i in range(100):
    signal = next(signal_source_items)

    traced_signal.trace_(signal)

    td_signal: SignalGridSet = SignalGridSet.temporal_diff(
        current=traced_signal, previous=previous_traced_signal
    )

    td_signal_trace.trace_(td_signal)
    if i > 95:
        print(f"Iteration {i}")
        print("Signal")
        signal.imshow()
        print("Signal trace")
        traced_signal.imshow()
        print("Previous signal trace")
        previous_traced_signal.imshow()
        print("Temporal diff of signal trace")
        td_signal.imshow()
        print("Trace of temporal diff")
        td_signal_trace.imshow()

        print("=" * 40)

    previous_traced_signal = traced_signal.clone()

td_signal_trace.imshow()

# %%
traced_signal.components["mu"].pixels.epsilon
# %%
