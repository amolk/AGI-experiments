import pytest

from patternmachine.clock import Clock


@pytest.fixture
def clock():
    return Clock(tau=2)


def test_clock_init(clock):
    assert clock is not None


def test_clock_tick_noop(clock):
    clock.tick()
    assert True


def test_iterator(clock):
    class TestIterator:
        def __init__(self):
            self.value = None
            self.iterator = iter(range(0, 10))

        def next(self, clock: Clock):
            self.value = next(self.iterator)
            return self.current(clock=clock)

        def current(self, clock: Clock):
            return self.value

    iterator = TestIterator()
    assert iterator.current(clock=clock) is None

    clock.register(iterator)
    assert iterator.current(clock=clock) is None

    clock.tick()
    assert iterator.current(clock=clock) is not None
    assert iterator.current(clock=clock) == 0

    clock.tick()
    assert iterator.current(clock=clock) is not None
    assert iterator.current(clock=clock) == 1
