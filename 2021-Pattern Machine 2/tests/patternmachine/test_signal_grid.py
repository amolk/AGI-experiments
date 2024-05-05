import pytest
import torch

from patternmachine.signal_grid import SignalGrid, SignalGridHP


def test_signal_grid_invalid_grid_size1():
    with pytest.raises(ValueError):
        SignalGridHP(grid_shape=(0, 4), signal_shape=(5, 6, 2))  # <-- zero


def test_signal_grid_invalid_grid_size2():
    with pytest.raises(ValueError):
        SignalGridHP(grid_shape=(1, 4), signal_shape=(5, -1, 2))  # <-- negative


@pytest.fixture
def signal_grid_hp1():
    return SignalGridHP(grid_shape=(3, 4), signal_shape=(5, 6, 2))


@pytest.fixture
def signal_grid_hp2():
    return SignalGridHP(grid_shape=(3, 4), signal_shape=(2, 2))


@pytest.fixture
def signal_grid1(signal_grid_hp1):
    return SignalGrid(hp=signal_grid_hp1)


def test_create_signal_grid(signal_grid_hp1):
    signal_grid = SignalGrid(hp=signal_grid_hp1)
    assert signal_grid.pixels.shape == (3 * 4, 5 * 6 * 2)
    assert signal_grid.precision.shape == signal_grid.pixels.shape
    assert torch.all(signal_grid.precision == 1.0) # default precision 1.0

def test_create_signal_grid_init_values(signal_grid_hp1):
    signal_grid = SignalGrid(hp=signal_grid_hp1, init_pixel_value=0.5, init_precision_value=0.2)
    assert torch.all(signal_grid.pixels == 0.5)
    assert torch.all(signal_grid.precision == 0.2)

def test_create_signal_grid_init_none(signal_grid_hp1):
    signal_grid = SignalGrid(hp=signal_grid_hp1, alloc_pixels=False, alloc_precision=False)
    assert signal_grid.pixels is None
    assert signal_grid.precision is None

@pytest.fixture
def signal_grid_hp_degenerate():
    return SignalGridHP(grid_shape=(1, 1), signal_shape=(1, 1))

def test_create_signal_grid_degenerate(signal_grid_hp_degenerate):
    signal_grid = SignalGrid(hp=signal_grid_hp_degenerate)
    assert signal_grid.pixels.shape == (1, 1)
    assert signal_grid.precision.shape == (1, 1)

def test_mix(signal_grid_hp_degenerate):
    test_cases = [
        # sg1.pixel, sg1.precision, sg2.pixel, sg2.precision, result.pixel, result.precision
        # [1, 1, 0, 0, 1, 1],  # 0 precision sg2 should be no-op
        # [0.5, 1, 0.5, 1, 0.5, 1], # same pixels, full precision => full precision (no uncertainty because sg1==sg2)
        # [1, 0.5, 1, 0.7, 1, 0.7], # same pixels => exp add precision
        # [0.4, 1, 0.6, 1, 0.5, 1.0], # similar pixels, full precision => avg pixel, high precision (low uncertainty because pixels similar)
        # [0.4, 0.5, 0.6, 0.6, 0.5, 0.6], # similar pixels, diff precision => avg pixel, exp add precision inv weighted by pixel diff
        [1, 1, 0, 1, 0.5, 0], # very diff pixels, full precision sg1 and sg2 => avg pixel, low precision (high uncertainty)
        [1, 0, 0, 0, 0, 0], # very diff pixels, zero precision sg1 and sg2 => n/a pixel, low precision (high uncertainty)
    ]

    for test_case in test_cases:
        sg1 = SignalGrid(hp=signal_grid_hp_degenerate, init_pixel_value=test_case[0], init_precision_value=test_case[1])
        sg2 = SignalGrid(hp=signal_grid_hp_degenerate, init_pixel_value=test_case[2], init_precision_value=test_case[3])
        result = SignalGrid.precision_weighted_mix([sg1, sg2])
        print("test_case", test_case)
        print("result.pixels", result.pixels)
        print("result.precision", result.precision)
        assert torch.isclose(result.pixels, torch.ones_like(result.pixels) * test_case[4], atol=1e-2)
        assert torch.isclose(result.precision, torch.ones_like(result.pixels)* test_case[5], atol=1e-2)
