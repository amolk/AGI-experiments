import torch
from config import Config

from patternmachine.clock import Clock


class Trace(torch.Tensor):
    # @classmethod
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     return super().__torch_function__(func, types, args, kwargs)

    @staticmethod
    def __new__(cls, x, epsilon=0.7, *args, **kwargs):
        return super().__new__(cls, x.float(), *args, **kwargs)

    def __init__(self, x, epsilon=0.7):
        self.epsilon = epsilon

    def clone(self, *args, **kwargs):
        return Trace(super().clone(*args, **kwargs), self.epsilon)

    def to(self, *args, **kwargs):
        new_obj = Trace([], self.epsilon)
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return new_obj

    """
    trace_()
    When adding a new tensor -
    Values updated as decaying history"""

    def trace_1(self, new_point_value: torch.Tensor):
        if not hasattr(self, "epsilon"):
            self.epsilon = 0.5

        diff = self.data - Config.BASE_ACTIVATION
        diff_sign = diff > 0

        data_contrast = (diff_sign * diff.abs() / (1 - Config.BASE_ACTIVATION)) + (
            diff_sign.logical_not_() * diff.abs() / Config.BASE_ACTIVATION
        )
        d = data_contrast * (1 - self.epsilon)
        self.data = new_point_value + (self.data - new_point_value) * d
        self.data.clamp_(min=0, max=1)

    def trace_(self, new_point_value: torch.Tensor):
        if not hasattr(self, "epsilon"):
            self.epsilon = 0.5

        self.data = self.data * self.epsilon + new_point_value * (1 - self.epsilon)
        self.data.clamp_(min=0, max=1)

    """
    ride_()
    When adding a new tensor -
    Values decay by (1-epsilon).
    Values go up quickly to higher values."""

    def ride_(self, new_point_value: torch.Tensor):
        self.data = self.data * (1 - self.epsilon)
        mask = self.data < new_point_value
        self.data[mask] = new_point_value[mask]

    """
    ride_inverse_()
    When adding a new tensor -
    Values grow toward 1.0 by (1+epsilon).
    Values go up quickly to lower values."""

    def ride_inverse_(self, new_point_value: torch.Tensor):
        self.data = self.data * (1 + self.epsilon)
        self.data.clamp_max_(1.0)
        mask = self.data > new_point_value
        self.data[mask] = new_point_value[mask]

    # def __repr__(self):
    # return f"Trace({self.epsilon}): " + self.shape.__repr__() + " " + super().__repr__()


class TraceIterator:
    def __init__(self, iterator, epsilon=0.7):
        assert iterator is not None
        assert hasattr(iterator, "next")
        assert hasattr(iterator, "current")

        self.iterator = iterator
        self.epsilon = epsilon
        self.trace = None

    def reset(self):
        self.trace = None

    def current(self, clock: Clock):
        return self.trace

    def next(self, clock: Clock):
        """Adds current value of iterator to trace.
        Note: Does NOT advance the iterator"""
        value = self.iterator.current
        print("value", value)
        assert isinstance(value, torch.Tensor)

        if self.trace is None:
            self.trace = Trace(value, epsilon=self.epsilon)
        else:
            self.trace.trace_(value)

        return self.trace
