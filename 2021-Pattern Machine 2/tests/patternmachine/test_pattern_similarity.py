import torch

from patternmachine.pattern_grid import PatternGrid, PatternGridHP
from patternmachine.pattern_similarity import PatternSimilarity, PatternSimilarityHP
from patternmachine.signal_grid import SignalGridHP
from patternmachine.signal_grid_set import SignalGridSet, SignalGridSetHP


def make_signal():
    hps = {
        "one": SignalGridHP(grid_shape=(2, 4), signal_shape=(3, 4)),
        "two": SignalGridHP(grid_shape=(2, 4), signal_shape=(3, 2, 1)),
    }
    pg_hp = SignalGridSetHP(hps=hps)
    return SignalGridSet(hp=pg_hp)


def make_patterns():
    pg_hp = PatternGridHP(
        grid_shape=(2, 4, 3, 2),
        pattern_signal_set_shape={"one": (3, 4), "two": (3, 2, 1)},
    )
    return PatternGrid(hp=pg_hp)


def test_sim_low_precision():
    # By default, very low precision, so equal even when pixels are random
    pgs = [make_signal(), make_patterns()]

    # force low precisions
    for _, component in pgs[1].begin.components.items():
        component.precision = torch.zeros_like(component.precision)

    sim = PatternSimilarity(
        signal=pgs[0],
        patterns=pgs[1].begin
    )
    assert sim.sim.shape == pgs[1].hp.grid_size
    assert torch.allclose(sim.sim, torch.ones_like(sim.sim))


def test_sim_disable_precision_weighting():
    # If disabled precision weighting, then unequal because pixels are random
    pgs = [make_signal(), make_patterns()]
    hp = PatternSimilarityHP(enable_precision_weighted_distance=False)
    sim = PatternSimilarity(
        signal=pgs[0],
        patterns=pgs[1].begin,
        hp=hp,
    )
    assert sim.sim.shape == pgs[1].hp.grid_size
    assert not torch.allclose(sim.sim, torch.ones_like(sim.sim))


def test_sim_high_precision():
    # If high precision, then unequal given pixels are random
    pgs = [make_signal(), make_patterns()]

    # force high precisions
    for _, component in pgs[1].begin.components.items():
        component.precision = torch.ones_like(component.precision)

    sim = PatternSimilarity(
        signal=pgs[0],
        patterns=pgs[1].begin
    )
    assert sim.sim.shape == pgs[1].hp.grid_size
    assert not torch.allclose(sim.sim, torch.ones_like(sim.sim))
