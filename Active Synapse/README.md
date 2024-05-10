# Active synapse based learning

This project explores an active dendrite based model for learning sparse representations of input patterns. The model is inspired by the ideas proposed in {Hawkins2016} and aims to demonstrate how specific dendrites on specific neurons in specific cortical columns become strongly connected to represent particular input patterns.

The model consists of a set of columns, each with a number of neurons, and each neuron having a set of dendrites with potential synapses. Binary input vectors with a sparsity of around 20% are used to train the model. Over repeated presentations of the inputs, the synaptic weights are strengthened for those dendrites and synapses that contribute to recognizing a particular input pattern.

The results show that after training, each sparse input pattern activates a specific set of dendrites on certain neurons in certain columns. Visualizing the receptive field weights reveals that the synapses contributing to the activation of a dendrite develop strong weights, while other synapses are weakened. 
