import nengo
from nengo.dists import Uniform

def negative(x):
    return -x

model = nengo.Network()
with model:
    stim = nengo.Node([0.2])
    input_rate_coding = nengo.Ensemble(
        1,
        dimensions=1,  # Represent a scalar
        # Set intercept to 0
        intercepts=Uniform(0, 0),
        # Set the maximum firing rate of the neuron to 100hz
        max_rates=Uniform(100, 100),
        # Set the neuron's firing rate to increase for positive input
        encoders=[[1]],
    )
    nengo.Connection(stim, input_rate_coding)

    l1 = nengo.Ensemble(
        1,
        dimensions=1,  # Represent a scalar
        # Set intercept to 0
        intercepts=Uniform(0, 0),
        # Set the maximum firing rate of the neuron to 100hz
        max_rates=Uniform(100, 100),
        # Set the neuron's firing rate to increase for positive input
        encoders=[[1]],
    )

    nengo.Connection(input_rate_coding, l1)
    nengo.Connection(l1, l1, function=negative)