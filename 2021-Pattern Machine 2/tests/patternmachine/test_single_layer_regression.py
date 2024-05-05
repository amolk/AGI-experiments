# import torch

# from patternmachine.single_layer_regression import SingleLayerRegression


# def test_single_layer_regression():
#     X = torch.linspace(0, 1, 100).unsqueeze(dim=-1)
#     Y = torch.linspace(0, 1, 100).unsqueeze(dim=-1)

#     pattern_machine = SingleLayerRegression(X, Y)
#     pattern_machine.layer.debug = False

#     # preset patterns with x and y as (0.0, 0.0) to (1.0, 1.0), i.e. y=x
#     for csgb in [
#         pattern_machine.layer.patterns.pixels_begin,
#         pattern_machine.layer.patterns.pixels_end,
#     ]:
#         csgb.components["x"].pixels = torch.linspace(0, 1, 10).unsqueeze(1)
#         csgb.components["y"].pixels = torch.linspace(0, 1, 10).unsqueeze(1)

#     # present pattern precision 1.0
#     for csgb in [
#         pattern_machine.layer.patterns.precision_begin,
#         pattern_machine.layer.patterns.precision_end,
#     ]:
#         csgb.components["x"].pixels = torch.ones_like(csgb.components["x"].pixels)
#         csgb.components["y"].pixels = torch.ones_like(csgb.components["y"].pixels)

#     print(
#         "pattern_machine.layer.patterns.composite_signal_grid_begin",
#         pattern_machine.layer.patterns.pixels_begin.pixels,
#     )
#     rms_error, mean_precision = pattern_machine.epoch()

#     assert rms_error < 0.03
#     assert mean_precision > 0.9
