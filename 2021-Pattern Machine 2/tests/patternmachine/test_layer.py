import torch

from patternmachine.layer import Layer, LayerHP
from patternmachine.signal_grid_set import SignalGridSet


def test_layer_create():
    hp = LayerHP(
        input_signal_shapes={"one": (10, 15), "two": (20, 10)},
        input_coverage_factor=1.0,
        patch_grid_shape=(5, 5),
        per_patch_pattern_grid_shape=(4, 4),
        output_patch_neighborhood_shape=(3, 3),
        output_decay=0.5,
    )
    layer = Layer(hp=hp)
    # pretty_print("Layer", layer)
    assert True


def test_layer_forward():
    input_signal_shapes = {"one": (10, 15), "two": (20, 10)}
    input_pixels_list = {
        name: torch.ones(shape).view(1, -1) for name, shape in input_signal_shapes.items()
    }
    input_precision_list = {
        name: torch.ones(shape).view(1, -1) for name, shape in input_signal_shapes.items()
    }
    input = SignalGridSet.from_pixels_and_precisions_list(
        pixels_list=input_pixels_list,
        precisions_list=input_precision_list,
        signal_shape=input_signal_shapes,
    )

    hp = LayerHP(
        input_signal_shapes=input_signal_shapes,
        input_coverage_factor=1.0,
        patch_grid_shape=(5, 5),
        per_patch_pattern_grid_shape=(3, 7),
        output_patch_neighborhood_shape=(3, 3),
        output_decay=0.5,
        pattern_init_pixel_values={"one": 1.0, "two": 1.0, "__output__": 0.0},
        pattern_init_precision_values={"one": 0.01, "two": 0.01, "__output__": 0.0},
    )

    layer = Layer(hp=hp)
    # pretty_print("Layer", layer)

    # print("input.pixels", input.pixels)
    # print("layer.patterns.begin.pixels", layer.patterns.begin.pixels)

    print("=" * 80)
    print("input")
    print(input.pixels)
    layer.forward(input)
    print("output")
    print(layer.output.pixels)
    assert torch.allclose(layer.output.pixels, torch.zeros_like(layer.output.pixels), atol=0.01)

    print("=" * 80)
    print("input")
    print(input.pixels)
    layer.forward(input)
    # print(layer.output.pixels)
    print("output")
    print(layer.output.pixels)
    assert torch.allclose(layer.output.pixels, torch.zeros_like(layer.output.pixels), atol=0.01)
