import pytest

from patternmachine.pattern_grid import PatternGrid, PatternGridHP


@pytest.fixture
def pg1():
    pg_hp = PatternGridHP(
        grid_shape=(1, 2), pattern_signal_set_shape={"one": (3, 4), "two": (3, 2, 1)}
    )

    return PatternGrid(hp=pg_hp)


def test_pg_create(pg1):
    pg1.alpha.shape == (1, 2)


@pytest.fixture
def pg2():
    pg_hp = PatternGridHP(
        grid_shape=(1,),
        pattern_signal_set_shape={
            "one": (1,),
            "two": (1,),
        },
    )

    return PatternGrid(hp=pg_hp)


def test_pg_create_2(pg2):
    pg2.alpha.shape == (1,)
