# Various experiments related to AGI

How would we build an AGI agent? Fundamentally there are two approaches. 
- Bottom-Up approachs would bring together models, algorithms, modules with specific capabilities and attempt to put those together in a composite AGI system.
- On the other hand, top-Down approaches attempt to build the simplest and least capable AGI agent first and then iterate to add capabilities.

I am interested in investigating Top-Down approaches to AGI.

How would we describe such a "minimal" AGI system? What capabilities would it have? How would we recognize it as an AGI? Would there be a single fundamental algorithm? What properties should such algorithm have? How would that algorithm produce an agent that exhibits representation learning and reinforcement learning?

I have been exploring these questions for a while now. This repository contains some of the investigations.

A few interesting notebooks -
- Representing values as histogram [obviates the need for precision weighting](https://github.com/amolk/AGI-experiments/blob/master/Free%20Energy%20Minimization%20Framework/12/13.ipynb).
- Can we learn latent variable probability distributions directly? This notebook explores the [Quantized Distribution Auto Encoder](https://github.com/amolk/AGI-experiments/blob/master/QDL%20-%20Quantized%20Distribution%20Learning/04.ipynb) approach.
- [Attractor learning](https://github.com/amolk/AGI-experiments/blob/master/Attractor%20Learning/03.ipynb): Imagine a neural network model that is allowed to settle its activity over time through lateral connection feedback. Can we train the network to achieve settled activity pattern quicker? This is along the lines of LISSOM, but using backprop so we can use DNN toolset.
- [VAE convolution kernel](https://github.com/amolk/AGI-experiments/blob/master/Autoencoding%20kernel%20convolution/12%20AE%20Kernel%20Convolutional%20Network%20(weighted%2C%20gaussian)%2C%20more%20layers%2C%20save%20to%20Drive.ipynb): What would happen if we used VAEs as convolution kernel? This notebook explores building 4 layer network that attempts to build a small top level latent representation of MNIST digits. Each layer is trained both independently (like DBN) and with top-down feedback.
- [Domain quantization / Information density normalization](https://github.com/amolk/AGI-experiments/blob/master/Domain%20Quantization/03%20-%20Domain%20quantization%203.ipynb): Each input is represented as a histogram. The model then adjusts the bins such that they are closely packed where many data points are present, i.e. precision follows information density. The model transforms input such that if input follows the learned distribution, the output is piecewise linear, which might help downstream layers to learn better.
- What is a good looking distribution? - [This notebook](https://github.com/amolk/AGI-experiments/blob/master/Free%20Energy%20Minimization%20Framework/12/16.1%20calculate%20precision%20of%20scaled%20pdf.ipynb) explores metrics to quantify if a distribution is good looking, i.e. has maximum signal to noise ratio.
- [Active dendrite models](https://github.com/amolk/AGI-experiments/blob/master/Active%20Synapse/01%20active%20synapse.ipynb): This script explores how an active dendrite based model could produce sparse latent representation of inputs.
