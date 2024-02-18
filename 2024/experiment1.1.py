# %%
import matplotlib.pyplot as plt
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import spikegen
from snntorch import spikeplot as splt


# %%
class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        num_neurons = 1
        beta = 0.9

        self.lif1 = snn.Leaky(beta=beta, reset_mechanism="zero", output=True)
        self.lif_inh2 = snn.Leaky(beta=beta, reset_mechanism="zero", output=True)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif_inh2.init_leaky()
        spk2 = torch.zeros_like(mem2)

        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []

        for step in range(x.size(0)):  # time x batch x num_inputs
            cur1 = x[step].unsqueeze(0)
            if spk2.shape[0] > 0:
                spk1, mem1 = self.lif1(-spk2 * 0.5, mem1)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2, mem2 = self.lif_inh2(spk1, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return (
            torch.stack(spk1_rec, dim=0),
            torch.stack(mem1_rec, dim=0),
            torch.stack(spk2_rec, dim=0),
            torch.stack(mem2_rec, dim=0),
        )


# %%
# %%
def plot_spikes(spike_data):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    splt.raster(spike_data, ax)
    plt.show()


def plot_voltage(mem_data):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(mem_data)
    plt.show()


# %%

# %%
from snntorch import spikegen

num_steps = 30
spike_data = spikegen.rate(torch.tensor([1.0]), num_steps=num_steps)
layer = Layer()

plot_spikes(spike_data)
spk1, mem1, spk2, mem2 = layer(spike_data * 0.3)
plot_voltage(mem1.view(-1).detach())
plot_spikes(spk1.view(-1).detach())
plot_voltage(mem2.view(-1).detach())
plot_spikes(spk2.view(-1).detach())
# %%
