import pytest
from pytest import approx
import torch

from patternmachine.clock import Clock
from patternmachine.trace import Trace, TraceIterator


@pytest.fixture
def trace():
    return Trace([1, 2, 3], epsilon=0.1)


def test_trace_init(trace):
    assert trace is not None


def test_trace_append(trace):
    trace.trace_(torch.Tensor([2, 3, 4]))
    assert trace[0].item() == approx(2)
    assert trace[1].item() == approx(3)
    assert trace[2].item() == approx(4)


# init Trace with a tensor
def test_trace_append2():
    trace = Trace(torch.Tensor([1, 2, 3]), epsilon=0.1)
    trace.trace_(torch.Tensor([2, 1, 3]))
    assert trace[0].item() == approx(2.0)  # increases quickly
    assert trace[1].item() == approx(1.8)  # decays slowly
    assert trace[2].item() == approx(3)  # stayed same


# test multiple append
def test_trace_append3(trace):
    trace = Trace(torch.Tensor([0, 0, 0, 0, 0]), epsilon=0.2)
    assert trace.allclose(torch.Tensor([0, 0, 0, 0, 0]))

    trace.trace_(torch.Tensor([0, 1, 0, 0, 0]))
    assert trace.allclose(torch.Tensor([0, 1, 0, 0, 0]))

    trace.trace_(torch.Tensor([0, 0, 1, 0, 0]))
    assert trace.allclose(torch.Tensor([0, 0.8, 1.0, 0, 0]))

    trace.trace_(torch.Tensor([0, 0, 0, 1, 0]))
    assert trace.allclose(torch.Tensor([0, 0.64, 0.8, 1.0, 0]))

    trace.trace_(torch.Tensor([0, 1, 0.7, 0, 1]))
    assert trace.allclose(torch.Tensor([0, 1.0, 0.7, 0.8, 1.0]))


def test_iterator_trace():
    class TestIterator:
        def __init__(self):
            self.value = None
            self.iterator = iter(
                [
                    torch.Tensor([0, 0, 0, 0, 0]),
                    torch.Tensor([0, 1, 0, 0, 0]),
                    torch.Tensor([0, 0, 1, 0, 0]),
                    torch.Tensor([0, 0, 0, 1, 0]),
                    torch.Tensor([0, 1, 0.7, 0, 1]),
                ]
            )

        def next(self, clock: Clock):
            self.value = next(self.iterator)
            return self.current

        @property
        def current(self):
            return self.value

    clock = Clock(tau=4)
    iterator = TestIterator()
    it = TraceIterator(iterator=iterator, epsilon=0.2)

    assert it is not None
    assert it.current(clock=clock) is None

    with pytest.raises(AssertionError):
        it.next(clock=clock)

    iterator.next(clock=clock)
    assert it.next(clock=clock) is not None
    assert it.current(clock=clock).allclose(torch.Tensor([0, 0, 0, 0, 0]))

    # no auto-advancing iterator
    it.next(clock=clock)
    assert it.current(clock=clock).allclose(torch.Tensor([0, 0, 0, 0, 0]))

    iterator.next(clock=clock)
    it.next(clock=clock)
    assert it.current(clock=clock).allclose(torch.Tensor([0, 1, 0, 0, 0]))

    iterator.next(clock=clock)
    it.next(clock=clock)
    assert it.current(clock=clock).allclose(torch.Tensor([0, 0.8, 1.0, 0, 0]))

    iterator.next(clock=clock)
    it.next(clock=clock)
    assert it.current(clock=clock).allclose(torch.Tensor([0, 0.64, 0.8, 1.0, 0]))

    iterator.next(clock=clock)
    it.next(clock=clock)
    assert it.current(clock=clock).allclose(torch.Tensor([0, 1.0, 0.7, 0.8, 1.0]))
