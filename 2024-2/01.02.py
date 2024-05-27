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


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


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
    assert len(configs) <= nrows * ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))
    if len(configs) == 1:
        axs = [axs]
    else:
        axs = fig.get_axes()

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
        target_frequency=0.1,
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
    def __init__(self, shape=(5, 5), pattern="square", noise=0.01):
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

        self.signal = self.signal * 0.95 + 0.05
        # self.spike_data = spikegen.rate(self.signal * 0.5, num_steps=self.num_steps)
        # self.t = 0
        print(self.signal)

    def forward(self):
        return self.signal + ((torch.rand(self.shape) * 2.0) - 1.0) * self.noise


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
    def __init__(self, experiment_name):
        super(Network, self).__init__()
        self.experiment_name = experiment_name
        create_folder_if_not_exists(f"output/{self.experiment_name}")
        self.sensory_input_shape = (5, 5)
        self.sensory_input = SensoryInput(
            shape=self.sensory_input_shape, pattern="border"
        )

        self.l1 = Ensemble(
            shape=self.sensory_input_shape,
            lateral_connections=False,
            base_threshold=1.0,
        )

    def forward(self, time_steps=100):
        history = defaultdict(list)

        for step in range(time_steps):
            x = self.sensory_input()
            history["sensory_input"].append(x)

            x = self.l1(x)

            history["l1_spikes"].append(x.clone())
            history["l1_activation"].append(self.l1.activation.clone())
            history["l1_threshold"].append(self.l1.threshold.clone())
            history["l1_frequency"].append(self.l1.frequency.activation.clone())

        for k in history:
            history[k] = torch.stack(history[k], dim=0).view(len(history[k]), -1)

        # Assuming `history` is the dictionary containing recorded history
        time_steps = history["l1_frequency"].shape[0]
        average_frequency = history["l1_frequency"].mean(dim=1)

        plt.figure(figsize=(10, 5))
        plt.plot(range(time_steps), average_frequency)
        plt.xlabel("Time Step")
        plt.ylabel("Average Frequency")
        plt.title("Average Neuron Firing Frequency Over Time")
        plt.savefig(f"output/{self.experiment_name}/average_frequency.png")
        plt.show()

        with open(f"output/{self.experiment_name}/video.html", "w") as f:
            animation_configs = [
                AnimatorConfig(
                    "Sensory input",
                    history["sensory_input"].view(
                        -1, self.sensory_input_shape[0], self.sensory_input_shape[1]
                    ),
                    0.0,
                    1.0,
                ),
                AnimatorConfig(
                    "L1 spikes",
                    history["l1_spikes"].view(
                        -1, self.sensory_input_shape[0], self.sensory_input_shape[1]
                    ),
                    0.0,
                    1.0,
                ),
                AnimatorConfig(
                    "L1 activation",
                    history["l1_activation"].view(
                        -1, self.sensory_input_shape[0], self.sensory_input_shape[1]
                    ),
                    0.0,
                    1.0,
                ),
                AnimatorConfig(
                    "L1 threshold",
                    history["l1_threshold"].view(
                        -1, self.sensory_input_shape[0], self.sensory_input_shape[1]
                    ),
                    0.0,
                    1.0,
                ),
                AnimatorConfig(
                    "L1 frequency",
                    history["l1_frequency"].view(
                        -1, self.sensory_input_shape[0], self.sensory_input_shape[1]
                    ),
                    0.0,
                    1.0,
                ),
            ]
            anim = animator(
                animation_configs,
                interval=100,
                nrows=3,
                ncols=2,
            )

            f.write(anim.to_jshtml())
        print("Done")


# %%
experiment_name = "01.01"
net = Network(experiment_name)
net(200)

# %%
