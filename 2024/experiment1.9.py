# %%
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import snntorch as snn
import snntorch.spikeplot as splt
import torch
import torch.nn as nn
from matplotlib import animation
from snntorch import spikegen
from snntorch import spikeplot as splt


def plot_spikes(spike_data, title="Spikes"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # set the title
    ax.set_title(title)
    splt.raster(spike_data, ax)
    plt.show()


def plot_voltage(mem_data, title="Membrane potential"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # set the title
    ax.set_title(title)
    ax.plot(mem_data)
    plt.show()


AnimatorConfig = namedtuple("AnimatorConfig", ["title", "data", "vmin", "vmax"])


# /Users/amolk/.pyenv/versions/3.10.5/envs/2024/lib/python3.10/site-packages/snntorch/spikeplot.py
# Changes:
#   - Added vmin and vmax parameters to animator
def animator(
    configs: list[AnimatorConfig],
    nrows=1,
    ncols=1,
    num_steps=False,
    interval=40,
    cmap="plasma",
):
    """Generate an animation by looping through the first dimension of a
    sample of spiking data.
    Time must be the first dimension of ``data``.

    Example::

        import snntorch.spikeplot as splt
        import matplotlib.pyplot as plt

        #  spike_data contains 128 samples, each of 100 time steps in duration
        print(spike_data.size())
        >>> torch.Size([100, 128, 1, 28, 28])

        #  Index into a single sample from a minibatch
        spike_data_sample = spike_data[:, 0, 0]
        print(spike_data_sample.size())
        >>> torch.Size([100, 28, 28])

        #  Plot
        fig, ax = plt.subplots()
        anim = splt.animator(spike_data_sample, fig, ax)
        HTML(anim.to_html5_video())

        #  Save as a gif
        anim.save("spike_mnist.gif")

    :param data: Data tensor for a single sample across time steps of
        shape [num_steps x input_size]
    :type data: torch.Tensor

    :param fig: Top level container for all plot elements
    :type fig: matplotlib.figure.Figure

    :param ax: Contains additional figure elements and sets the coordinate
        system. E.g.:
            fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    :type ax: matplotlib.axes._subplots.AxesSubplot

    :param num_steps: Number of time steps to plot. If not specified,
        the number of entries in the first dimension
            of ``data`` will automatically be used, defaults to ``False``
    :type num_steps: int, optional

    :param interval: Delay between frames in milliseconds, defaults to ``40``
    :type interval: int, optional

    :param cmap: color map, defaults to ``plasma``
    :type cmap: string, optional

    :return: animation to be displayed using ``matplotlib.pyplot.show()``
    :rtype: FuncAnimation

    """
    assert len(configs) == nrows * ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    if len(configs) == 1:
        axs = [axs]

    # datas = [c.data.cpu() for c in configs]
    # vmins = [c.vmin for c in configs]
    # vmaxs = [c.vmax for c in configs]

    if not num_steps:
        num_steps = configs[0].data.size()[0]

    plt.axis("off")

    artists = []

    for step in range(num_steps):
        artist = []
        for i in range(len(configs)):
            config = configs[i]
            axs[i].axis("off")
            axs[i].set_title(config.title)
            im = axs[i].imshow(
                config.data[step], cmap=cmap, vmin=config.vmin, vmax=config.vmax
            )
            artist.append(im)
        artists.append(artist)

    anim = animation.ArtistAnimation(
        fig, artists, interval=interval, blit=False, repeat_delay=1000
    )

    return anim


# Collector node (sums up all provided inputs)
class Collector(nn.Module):
    def __init__(self, shape=(5, 5)):
        super(Collector, self).__init__()
        self.shape = shape
        self.activation = torch.zeros(shape)

    def reset(self):
        self.activation = torch.zeros(self.shape)

    def forward(self, x):
        assert x.shape == self.shape
        self.activation += x


# Leaky integrator neuron (no firing)
class Leaky(nn.Module):
    def __init__(self, shape=(5, 5), alpha=0.2, beta=0.9):
        super(Leaky, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.shape = shape
        self.activation = torch.zeros(shape)

    def forward(self, x):
        assert x.shape == self.shape
        # assert x.dtype == torch.bool

        # decay
        self.activation = self.beta * self.activation

        # integrate (assympotically approach 1.0)
        self.activation += (1 - self.activation) * self.alpha * x


# Running average integrator neuron (no firing)
class RunningAverage(nn.Module):
    def __init__(self, shape=(5, 5), beta=0.95):
        super(RunningAverage, self).__init__()
        self.beta = beta
        self.shape = shape
        self.activation = torch.zeros(shape)

    def forward(self, x):
        assert x.shape == self.shape
        # assert x.dtype == torch.bool

        # decay
        self.activation = self.beta * self.activation + (1 - self.beta) * x


# Rider - Rapidly charges to input value and then slowly decays
class Rider(nn.Module):
    def __init__(self, shape=(5, 5), beta=0.9):
        super(Rider, self).__init__()
        self.beta = beta
        self.shape = shape
        self.activation = torch.zeros(shape)

    def forward(self, x):
        assert x.shape == self.shape
        # assert x.dtype == torch.bool

        # decay
        self.activation = self.beta * self.activation

        # ride any higher input
        self.activation = torch.max(self.activation, x)


class Ensemble(nn.Module):
    def __init__(
        self,
        shape=(5, 5),
        beta=0.9,
        base_threshold=1.0,
        target_frequency=0.2,
        reset_mechanism="zero",
        auto_gain_control=True,
        lateral_connections=True,
    ):
        super(Ensemble, self).__init__()
        self.shape = shape
        self.beta = beta
        self.base_threshold = base_threshold
        self.target_frequency = target_frequency
        self.reset_mechanism = reset_mechanism
        self.auto_gain_control = auto_gain_control
        self.lateral_connections = lateral_connections

        self.activation = torch.zeros(shape)
        self.spikes = torch.zeros(shape, dtype=torch.bool)
        if auto_gain_control:
            self.threshold = torch.ones(shape) * self.base_threshold
            self.frequency = RunningAverage(shape)
        else:
            self.threshold = self.base_threshold

        if lateral_connections:
            self.lateral_weights = torch.zeros(shape[0] * shape[1], shape[0] * shape[1])

    def forward(self, x):
        assert x.shape == self.shape

        # lateral input
        if self.lateral_connections:
            lateral_input = (
                self.lateral_weights[self.spikes.view(-1), :]
                .sum(dim=0)
                .view(self.shape)
            )
            x = x + lateral_input

        self.activation = self.beta * self.activation + x
        self.spikes = self.activation > self.threshold

        if self.auto_gain_control:
            self.frequency(self.spikes)
            self.threshold[self.frequency.activation > self.target_frequency] += 0.05
            self.threshold[self.frequency.activation < self.target_frequency] /= 1.05

        if self.reset_mechanism == "zero":
            self.activation[self.spikes] = 0.0
        elif self.reset_mechanism == "subtract":
            self.activation[self.spikes] -= self.threshold[self.spikes]
        else:
            raise ValueError("Reset mechanism not recognized")

        return self.spikes


class SensoryInput(nn.Module):
    def __init__(self, shape=(5, 5), pattern="square", noise=0.05):
        super(SensoryInput, self).__init__()
        self.shape = shape
        self.num_steps = 100
        self.noise = noise

        if pattern == "square":
            self.signal = torch.zeros(shape)
            self.signal[1:3, 1:3] = 1
        elif pattern == "corners":
            self.signal = torch.zeros(shape)
            self.signal[0, 0] = 1
            self.signal[0, -1] = 1
            self.signal[-1, 0] = 1
            self.signal[-1, -1] = 1
        elif pattern == "border":
            self.signal = torch.zeros(shape)
            self.signal[0, :] = 1
            self.signal[-1, :] = 1
            self.signal[:, 0] = 1
            self.signal[:, -1] = 1
        elif pattern == "checkerboard":
            self.signal = torch.zeros(shape)
            self.signal[::2, ::2] = 1
            self.signal[1::2, 1::2] = 1
        elif pattern == "L":
            self.signal = torch.zeros(shape)
            self.signal[:, 0] = 1
            self.signal[-1, :] = 1
        elif pattern == "mix":  # mix of square and border
            self.signal = torch.zeros(shape)
            self.signal[1:3, 1:3] = 1
            self.signal[0, :] = 1
            self.signal[-1, :] = 1
            self.signal[:, 0] = 1
            self.signal[:, -1] = 1
        elif pattern == "empty":
            self.signal = torch.zeros(shape)
        elif pattern == "random":
            self.signal = torch.rand(shape)

        self.spike_data = spikegen.rate(self.signal * 0.5, num_steps=self.num_steps)
        self.t = 0

    def forward(self):
        self.t += 1
        spikes = self.spike_data[self.t % self.num_steps].clone()

        # add some noise (set 5% pixels to 0 and 5% to 1)
        spikes[torch.rand(self.shape) < self.noise] = 0
        spikes[torch.rand(self.shape) < self.noise] = 1

        return spikes


class Connection(nn.Module):
    def __init__(self, from_shape=(5, 5), to_shape=(5, 5)):
        super(Connection, self).__init__()
        self.from_shape = from_shape
        self.to_shape = to_shape
        # weights are flattened from_shape by flattened to_shape
        self.weights = torch.rand(
            size=(from_shape[0] * from_shape[1], to_shape[0] * to_shape[1]),
        )
        self.weights = self.weights / self.weights.sum(dim=0)

    def forward(self, x):
        return (x.float().view(-1) @ self.weights).view(self.to_shape)


class Network(nn.Module):
    def __init__(
        self,
        prediction_mixed_input=True,
        prediction_precision_gain=0.1,
        lateral_connections=True,
    ):
        super(Network, self).__init__()
        self.prediction_mixed_input = prediction_mixed_input
        self.prediction_precision_gain = prediction_precision_gain
        self.lateral_connections = lateral_connections

        self.ensemble1_shape = (1, 2)
        self.sensory_input_shape = (5, 5)
        self.sensory_input = SensoryInput(
            shape=self.sensory_input_shape, pattern="square"
        )
        self.sensory_input2 = SensoryInput(shape=self.sensory_input_shape, pattern="L")
        self.sensory_input3 = SensoryInput(
            shape=self.sensory_input_shape, pattern="border"
        )
        self.sensory_input_mix = SensoryInput(
            shape=self.sensory_input_shape, pattern="mix"
        )
        self.sensory_input_empty = SensoryInput(
            shape=self.sensory_input_shape, pattern="empty"
        )
        self.ensemble1_input_collector = Collector(shape=self.sensory_input_shape)

        self.ensemble1 = Ensemble(
            shape=self.ensemble1_shape,
            lateral_connections=lateral_connections,
            base_threshold=0.1,
        )
        if lateral_connections:
            self.ensemble1.lateral_weights = torch.tensor([[0.0, -0.5], [-0.5, 0.0]])

        # connect collector to ensemble
        self.collector_to_ensemble_connection = Connection(
            from_shape=self.ensemble1_input_collector.shape,
            to_shape=self.ensemble1.shape,
        )
        # set first neuron to be sensitive to the sensory input 1
        pattern = (
            (self.sensory_input.signal / self.sensory_input.signal.norm(1))
        ).view(-1)
        # pattern[pattern < 0.01] -= 0.1
        self.collector_to_ensemble_connection.weights[:, 0] = pattern
        # set second neuron to be sensitive to sensory input 3
        pattern = (
            (self.sensory_input3.signal / self.sensory_input3.signal.norm(1))
        ).view(-1)
        # pattern[pattern < 0.01] -= 0.1
        self.collector_to_ensemble_connection.weights[:, 1] = pattern
        # self.collector_to_ensemble_connection.weights[:, 1] = (
        #     self.collector_to_ensemble_connection.weights[:, 1]
        #     / self.collector_to_ensemble_connection.weights[:, 1].abs().sum()
        # )

        if self.prediction_mixed_input:
            self.top_down_connection = Connection(
                from_shape=self.ensemble1.shape,
                to_shape=self.ensemble1_input_collector.shape,
            )

            # set first neuron's prediction to be same as its preferred input pattern
            self.top_down_connection.weights[0, :] = self.sensory_input.signal.view(-1)

            # set second neuron's prediction to be same as its preferred input pattern
            self.top_down_connection.weights[1, :] = self.sensory_input3.signal.view(-1)

    def forward(self, time_steps=100):
        history = defaultdict(list)
        smoothed_input = RunningAverage(shape=self.sensory_input_shape)
        sensory_input_rider = RunningAverage(shape=self.sensory_input_shape, beta=0.9)
        prediction_rider = RunningAverage(shape=self.sensory_input_shape, beta=0.9)

        for step in range(time_steps):
            self.ensemble1_input_collector.reset()

            # sensory input -> collector
            if step > time_steps * 4 // 5:
                x = self.sensory_input_mix() * 0.3  # uncertain mix
            elif step > time_steps * 3 // 5:
                x = self.sensory_input2()
            elif step > time_steps * 2 // 5:
                x = self.sensory_input_empty()
            elif step > time_steps * 1 // 5:
                x = self.sensory_input3()
            else:
                x = self.sensory_input()
            history["sensory_input"].append(x)
            sensory_input_rider(x)
            self.ensemble1_input_collector(x)

            # top-down prediction -> collector
            if self.prediction_mixed_input:
                prediction = self.top_down_connection(
                    self.ensemble1.spikes.float() * self.prediction_precision_gain
                )
                history["ensemble1_prediction"].append(prediction.clone())
                self.ensemble1_input_collector(prediction)

            # collector -> ensemble
            x = self.ensemble1_input_collector.activation
            history["ensemble1_input"].append(x.clone())
            smoothed_input(x)
            history["ensemble1_smoothed_input"].append(
                smoothed_input.activation.clone()
            )

            x = self.collector_to_ensemble_connection(x)
            x = self.ensemble1(x)

            if step == 700:
                a = 10

            prediction_rider(self.top_down_connection(self.ensemble1.spikes.float()))
            prediction_error = (
                sensory_input_rider.activation - prediction_rider.activation
            ).abs()  # * prediction_rider.activation

            history["ensemble1_spikes"].append(x.clone())
            history["ensemble1_activation"].append(self.ensemble1.activation.clone())
            history["ensemble1_threshold"].append(self.ensemble1.threshold.clone())
            history["ensemble1_frequency"].append(
                self.ensemble1.frequency.activation.clone()
            )
            history["sensory_input_rider"].append(
                sensory_input_rider.activation.clone()
            )
            history["prediction_rider"].append(prediction_rider.activation.clone())
            history["ensemble1_prediction_error"].append(prediction_error)
            history["ensemble1_prediction_error_average"].append(
                prediction_error.mean()
            )

        for k in history:
            history[k] = torch.stack(history[k], dim=0).view(len(history[k]), -1)

        plot_spikes(history["sensory_input"], title="Sensory input spikes")
        plot_voltage(history["ensemble1_activation"], title="Ensemble 1 activation")
        plot_spikes(history["ensemble1_spikes"], title="Ensemble 1 spikes")
        plot_voltage(history["ensemble1_threshold"], title="Ensemble 1 threshold")
        plot_voltage(history["ensemble1_frequency"], title="Ensemble 1 frequency")
        plot_voltage(
            history["ensemble1_prediction_error_average"],
            title="Ensemble 1 average prediction error",
        )

        with open("images/video.html", "w") as f:
            animation_configs = [
                AnimatorConfig(
                    "Sensory input rider",
                    history["sensory_input_rider"].view(
                        -1, self.sensory_input_shape[0], self.sensory_input_shape[1]
                    ),
                    0.0,
                    1.0,
                ),
                AnimatorConfig(
                    "Prediction rider",
                    history["prediction_rider"].view(
                        -1, self.sensory_input_shape[0], self.sensory_input_shape[1]
                    ),
                    0.0,
                    1.0,
                ),
                AnimatorConfig(
                    "Prediction error",
                    history["ensemble1_prediction_error"].view(
                        -1, self.sensory_input_shape[0], self.sensory_input_shape[1]
                    ),
                    0.0,
                    1.0,
                ),
            ]
            anim = animator(
                animation_configs,
                interval=100,
                nrows=len(animation_configs),
                ncols=1,
            )

            f.write(anim.to_jshtml())
        print("Done")


# %%
net = Network(
    prediction_mixed_input=True,
    prediction_precision_gain=0.25,
    lateral_connections=True,
)
net(2000)
# %%
