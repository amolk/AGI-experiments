# Pattern machine

*2 Feb 2024*
After doing a bunch of experiments with nanoGPT, I continue to see the need for spiking neurons as a way to convey evidence for a pattern, sparse population code to convey confidence/precision and network dynamics as a way to reconsile bottom up inputs with top down predictions. So, here I am, yet again starting experiments with spiking neurons as a way to build Pattern Machine.

Once again, I am going to attempt to capure the dynamics that, in my opinion, underlie many desirable properties of the brain and a future AGI system.

Three kinds of spike codes are discussed in the literature - rate code, latency and diff. I posit that these three codes are facets of the same underlying code. Yes, larger input current would make a neuron fire more, so rate code makes sense. At the same time, larger input current also makes neurons fire earlier, which represents latency code when we look at the first spike. Predictive coding inhibition would clamp down stable (predictable) neural activity and thus a neuron may appear to fire only on change. Again, just a facet of the same underlying neural code. I wonder whether I should write a paper to explore this point of view, so the field can move on from the debate about which code is the "correct" one.

Coming back to the desirable dynamics discussion, here is the dynamic I am going after. A neuron is driven by bottom up input and top down predictions where the influence of these two signals oscillate out of phase. Towards the start of a cycle, bottom up input drives the neuron and later on in the cycle, top down predictions drive it.

First experiment is to set up a single neuron that demonstrates this cyclic behavior.

Note: The prediction is be for the next time step, i.e. the neuron defines a transition function. Higher layers could have longer time steps and thus drive lower layers to recapituate future trajectories, useful for planning.

## experiment1.py
Tried with nengo. Realized that Node to Node connections are implemented as decoded value, not individual spikes, so not suitable.

