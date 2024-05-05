# %%
from IPython.display import display


print = display

import sys

sys.path.append("/Users/amolk/work/AGI/pattern-machine/src")

from patternmachine.signal_source.rotating_rectangle_signal_source import (
    RotatingRectangleSignalSource,
)
from patternmachine.signal_source.moving_rectangle_signal_source import (
    MovingRectangleSignalSource,
)
from patternmachine.signal_source.cached_frames_video_source import CachedFramesVideoSource

image_shape = (100, 100)

# %%
signal_source = RotatingRectangleSignalSource(
    height=image_shape[0], width=image_shape[1], angle_step=7
)
signal_source.imshow()

# %%
signal_source = CachedFramesVideoSource(height=image_shape[0], width=image_shape[1])
signal_source.imshow()

# %%

signal_source = MovingRectangleSignalSource(height=image_shape[0], width=image_shape[1])
signal_source.imshow(100)

# %%
