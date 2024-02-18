# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import spikegen
from snntorch import spikeplot as splt


def plot_spikes(spike_data):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    splt.raster(spike_data, ax)
    plt.show()


def plot_voltage(mem_data):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(mem_data)
    plt.show()


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
        assert x.dtype == torch.bool

        # decay
        self.activation = self.beta * self.activation

        # integrate (assympotically approach 1.0)
        self.activation += (1 - self.activation) * self.alpha * x


class Ensemble(nn.Module):
    def __init__(
        self, shape=(5, 5), beta=0.9, reset_mechanism="zero", auto_gain_control=True
    ):
        super(Ensemble, self).__init__()
        self.beta = beta
        self.reset_mechanism = reset_mechanism
        self.shape = shape
        self.auto_gain_control = auto_gain_control
        self.activation = torch.zeros(shape)
        self.spikes = torch.zeros(shape, dtype=torch.bool)
        if auto_gain_control:
            self.threshold = torch.ones(shape)
            self.frequency = Leaky(shape)
        else:
            self.threshold = 1.0

        self.lateral_weights = torch.zeros(shape[0] * shape[1], shape[0] * shape[1])

    def forward(self, x):
        assert x.shape == self.shape

        # lateral input
        lateral_input = (
            self.lateral_weights[self.spikes.view(-1), :].sum(dim=0).view(self.shape)
        )
        x = x + lateral_input

        self.activation = self.beta * self.activation + x
        self.spikes = self.activation > self.threshold

        if self.auto_gain_control:
            self.frequency(self.spikes)
            self.threshold = 1 + self.frequency.activation

        if self.reset_mechanism == "zero":
            self.activation[self.spikes] = 0.0
        elif self.reset_mechanism == "subtract":
            self.activation[self.spikes] -= self.threshold[self.spikes]
        else:
            raise ValueError("Reset mechanism not recognized")

        return self.spikes


class SensoryInput(nn.Module):
    def __init__(self, shape=(5, 5)):
        super(SensoryInput, self).__init__()
        self.shape = shape
        self.num_steps = 100
        self.signal = torch.rand(shape)
        self.spike_data = spikegen.rate(self.signal, num_steps=self.num_steps)
        self.t = 0

    def forward(self):
        self.t += 1
        return self.spike_data[self.t % self.num_steps]


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
        return (x.view(-1) @ self.weights).view(self.to_shape)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.shape1 = (5, 5)
        self.shape2 = (1, 2)
        self.sensory_input = SensoryInput(shape=self.shape1)
        self.sensory_input2 = SensoryInput(shape=self.shape1)

        self.ensemble1 = Ensemble(shape=self.shape2)
        self.ensemble1.lateral_weights = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])

        self.connection1 = Connection(
            from_shape=self.sensory_input.shape, to_shape=self.ensemble1.shape
        )

        # set first neuron to be sensitive to the sensory input
        self.connection1.weights[:, 0] = self.sensory_input.signal.view(-1)
        self.connection1.weights[:, 0] = (
            self.connection1.weights[:, 0] / self.connection1.weights[:, 0].sum()
        )
        # set second neuron to be sensitive to 1 - sensory input
        self.connection1.weights[:, 1] = (1 - self.sensory_input.signal).view(-1)
        self.connection1.weights[:, 1] = (
            self.connection1.weights[:, 1] / self.connection1.weights[:, 1].sum()
        )

    def forward(self, time_steps=100):
        history = defaultdict(list)

        for step in range(time_steps):
            x = self.sensory_input()
            if step > time_steps * 3 // 4:
                x = (torch.rand(self.shape1) > 0.5) * 1.0
            elif step > time_steps // 2:
                x = self.sensory_input2()
            elif step > time_steps // 4:
                x = 1 - x
            history["sensory_input"].append(x)
            x = self.connection1(x)
            x = self.ensemble1(x)
            history["ensemble1_spikes"].append(x)
            history["ensemble1_activation"].append(self.ensemble1.activation)
            history["ensemble1_threshold"].append(self.ensemble1.threshold)
            history["ensemble1_frequency"].append(self.ensemble1.frequency.activation)

        for k in history:
            history[k] = torch.stack(history[k], dim=0).view(len(history[k]), -1)

        plot_spikes(history["sensory_input"])
        # plot_voltage(history["ensemble1_activation"])
        plot_spikes(history["ensemble1_spikes"])
        # plot_voltage(history["ensemble1_threshold"])
        plot_voltage(history["ensemble1_frequency"])


# %%
net = Network()
net(900)
# %%
