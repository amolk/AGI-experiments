# %%
%cd ~/work/free-energy-minimization-framework/9/
%load_ext autoreload
%autoreload 2

# %%
from f import F
import torch
from torch import nn
import pdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
%matplotlib inline

figsize=(15,5)
learning_rate = 0.01
quantization = 4

# %%
pattern_length = 4
pattern = torch.zeros((pattern_length,quantization))
pattern[0][0] = 1
pattern[1][1] = 1
pattern[2][2] = 1
pattern[3][3] = 1

plt.figure(figsize=figsize)
plt.imshow(pattern.t().numpy(), cmap='gray', label='pattern 1')
plt.show()

pattern2 = pattern.clone()
pattern2[3][3] = 0
pattern2[3][0] = 1

plt.figure(figsize=figsize)
plt.imshow(pattern2.t().numpy(), cmap='gray', label='pattern 2')
plt.show()

input = torch.stack((pattern[:-1], pattern2[:-1], pattern2[:-1], pattern2[:-1]))
print(input.shape)
target = torch.stack((pattern[1:], pattern2[1:], pattern2[1:], pattern2[1:]))
batch_size = input.shape[0]
t_sample = input.shape[1]

# print('input', input, 'target', target)

hidden_size = t_sample
num_layers = 2
# %%
rnn = nn.RNN (
  input_size=quantization,
  hidden_size=quantization,
  num_layers=num_layers,
  nonlinearity='tanh',
  batch_first=True
)

# linear = nn.Linear (
#   hidden_size,
#   quantization
# )

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
epoch = 0
losses = []
while epoch < 300:
  rnn.zero_grad()
  state = torch.zeros(num_layers, quantization, quantization)
  out, state = rnn(input, state)
  # out = linear(out)
  # print('out', out)
  # print('state', state)
  loss = torch.nn.functional.mse_loss(out, target)
  loss.backward()
  optimizer.step()
  # print('loss', loss)
  losses.append(loss)
  epoch += 1

plt.figure(figsize=figsize)
plt.plot(losses, label='loss')
plt.legend()
plt.show()

# %%

state = torch.zeros(num_layers, quantization, quantization)
out, state = rnn(input, state)

plt.figure(figsize=figsize)
plt.imshow(out[1].detach().t(), cmap='gray', label='output')
plt.legend()
plt.show()

print("The last column shows the appropriate expected probability distribution [0.75, 0, 0, 0.25]")
print("becasue pattern 1 was shown 1 time out of 4 samples and pattern 2 was shown 3 times out of 4 samples.")

# %%
