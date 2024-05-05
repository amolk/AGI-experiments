import pytest
import torch

from patternmachine.signal_grid_set import (
    GridShapeMismatchError,
    NoComponentsError,
    SignalGrid,
    SignalGridHP,
    SignalGridSet,
    SignalGridSetHP,
)


@pytest.fixture
def signal_grid_hp1():
    return SignalGridHP(grid_shape=(3, 4), signal_shape=(5, 6, 2))


@pytest.fixture
def signal_grid_hp2():
    return SignalGridHP(grid_shape=(3, 4), signal_shape=(2, 2))


def test_signal_grid_set_from_pixels_list():
    csg = SignalGridSet.from_pixels_list({"one": torch.ones((10, 10)), "two": torch.ones((5, 5))})
    assert csg.grid_shape == (1, 1)
    assert len(csg.components) == 2

    c0 = csg.components["one"]
    assert c0.hp.grid_shape == (1, 1)
    assert c0.signal_shape == (10, 10)

    c1 = csg.components["two"]
    assert c1.hp.grid_shape == (1, 1)
    assert c1.signal_shape == (5, 5)


def test_signal_grid_set_from_pixels_list_and_signal_shapes():
    csg = SignalGridSet.from_pixels_list(
        pixels_list={"one": torch.ones((1, 100)), "two": torch.ones((25, 1))},
        signal_shape={"one": (10, 10), "two": (5, 5)},
    )
    assert csg.grid_shape == (1, 1)
    assert len(csg.components) == 2

    c0 = csg.components["one"]
    assert c0.hp.grid_shape == (1, 1)
    assert c0.signal_shape == (10, 10)

    c1 = csg.components["two"]
    assert c1.hp.grid_shape == (1, 1)
    assert c1.signal_shape == (5, 5)


def test_signal_grid_set_from_signal_grids_error1():
    grid_shape0 = (3, 4)
    signal_shape0 = (5, 3, 2)
    grid_shape1 = (1, 2)
    signal_shape1 = (12,)

    sgs = {
        "one": SignalGrid(hp=SignalGridHP(grid_shape=grid_shape0, signal_shape=signal_shape0)),
        "two": SignalGrid(hp=SignalGridHP(grid_shape=grid_shape1, signal_shape=signal_shape1)),
    }
    with pytest.raises(GridShapeMismatchError):
        SignalGridSet.from_signal_grids(sgs)


def test_signal_grid_set_from_signal_grids():
    grid_shape = (3, 4)
    signal_shape0 = (5, 3, 2)
    signal_shape1 = (12,)

    sgs = {
        "one": SignalGrid(hp=SignalGridHP(grid_shape=grid_shape, signal_shape=signal_shape0)),
        "two": SignalGrid(hp=SignalGridHP(grid_shape=grid_shape, signal_shape=signal_shape1)),
    }
    csg = SignalGridSet.from_signal_grids(sgs)

    assert csg.hp.grid_shape == grid_shape
    assert len(csg.components) == 2

    c0 = csg.components["one"]
    assert c0.hp.grid_shape == grid_shape
    assert c0.signal_shape == signal_shape0

    c1 = csg.components["two"]
    assert c1.hp.grid_shape == grid_shape
    assert c1.signal_shape == signal_shape1


# def test_signal_grid_set_from_pixels_and_variances_list():
#     pixels_list = {"one": torch.ones((10, 10)), "two": torch.ones((5, 5))}
#
#     csg = SignalGridSet.from_pixels_and_variances_list(
#         pixels_list=pixels_list, variances_list=variances_list, signal_shape=[(10, 10), (5, 5)]
#     )
#     assert csg.grid_shape == (1, 1)
#     assert len(csg.components) == 2
#
#     c0 = csg.components[0]
#     assert c0.hp.grid_shape == (1, 1)
#     assert c0.signal_shape == (10, 10)
#     assert c0.pixels is pixels_list[0]
#     assert c0.variance is variances_list[0]
#
#     c1 = csg.components[1]
#     assert c1.hp.grid_shape == (1, 1)
#     assert c1.signal_shape == (5, 5)
#     assert c1.pixels is pixels_list[1]
#     assert c1.variance is variances_list[1]


def test_signal_grid_set_indexing(signal_grid_hp1, signal_grid_hp2):
    csg_hp = SignalGridSetHP(hps={"one": signal_grid_hp1, "two": signal_grid_hp2})
    csg = SignalGridSet(hp=csg_hp)

    for index in range(5, 10):
        item = csg[index]
        assert item.hp.grid_shape == (1, 1)
        assert len(item.hp.components) == len(csg.hp.components)
        assert item.signal_shape == csg.signal_shape


def test_csg_zero_components():
    """
    Must have at least 1 component
    """
    with pytest.raises(NoComponentsError):
        SignalGridSetHP({})


def test_csg_different_grid_shapes():
    """
    Components may not have different grid shapes
    """
    with pytest.raises(GridShapeMismatchError):
        SignalGridSetHP(
            {
                "one": SignalGridHP(grid_shape=(1, 2), signal_shape=(3, 4)),
                "two": SignalGridHP(grid_shape=(2, 2), signal_shape=(3, 4)),
            }
        )


@pytest.fixture
def signal_grid_set1():
    return SignalGridSet(
        SignalGridSetHP(
            {
                "one": SignalGridHP(grid_shape=(1, 2), signal_shape=(3, 4)),
                "two": SignalGridHP(grid_shape=(1, 2), signal_shape=(3, 2, 1)),
            }
        )
    )


def test_csg_different_signal_shapes(signal_grid_set1):
    """
    Components may have different signal shapes
    """
    assert signal_grid_set1.component_count == 2
    assert signal_grid_set1.components["one"].signal_shape == (3, 4)
    assert signal_grid_set1.components["two"].signal_shape == (3, 2, 1)
    assert signal_grid_set1.signal_shape == {"one": (3, 4), "two": (3, 2, 1)}
