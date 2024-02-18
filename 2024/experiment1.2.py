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


class Ensemble(nn.Module):
    def __init__(self, shape=(5, 5), beta=0.9, reset_mechanism="zero"):
        super(Ensemble, self).__init__()
        self.beta = beta
        self.reset_mechanism = reset_mechanism
        self.shape = shape
        self.activation = torch.zeros(shape)
        self.spikes = torch.zeros(shape, dtype=torch.bool)

    def forward(self, x):
        assert x.shape == self.shape
        self.activation = self.beta * self.activation + x
        self.spikes = self.activation > 1.0

        if self.reset_mechanism == "zero":
            self.activation[self.spikes] = 0.0
        elif self.reset_mechanism == "subtract":
            self.activation[self.spikes] -= 1.0
        else:
            raise ValueError("Reset mechanism not recognized")

        return self.spikes


class SensoryInput(nn.Module):
    def __init__(self, shape=(5, 5)):
        super(SensoryInput, self).__init__()
        self.shape = shape
        self.num_steps = 100
        self.spike_data = spikegen.rate(torch.rand(shape), num_steps=self.num_steps)
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
        self.shape2 = (1, 1)
        self.sensory_input = SensoryInput(shape=self.shape1)
        self.ensemble1 = Ensemble(shape=self.shape2)
        self.connection1 = Connection(
            from_shape=self.sensory_input.shape, to_shape=self.ensemble1.shape
        )

    def forward(self):
        history = defaultdict(list)

        for _ in range(100):
            x = self.sensory_input()
            history["sensory_input"].append(x)
            x = self.connection1(x)
            x = self.ensemble1(x)
            history["ensemble1_spikes"].append(x)
            history["ensemble1_activation"].append(self.ensemble1.activation)

        for k in history:
            history[k] = torch.stack(history[k], dim=0).view(len(history[k]), -1)

        plot_spikes(history["sensory_input"])
        # plot_voltage(history["ensemble1_activation"])
        plot_spikes(history["ensemble1_spikes"])


# %%
net = Network()
net()
# %%
