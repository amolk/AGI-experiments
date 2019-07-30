# %%
%cd ~/work/free-energy-minimization-framework/7/
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
t_sample = 20
learning_rate = 0.001
# sin wave with noise present only certain ranges
pattern_length = 100
pattern = torch.tensor(np.sin(np.arange(pattern_length) * 0.30) - np.sin(np.arange(pattern_length) * 0.20) + np.random.sample(pattern_length) * 0.3).float().unsqueeze(1)
plt.figure(figsize=figsize)
plt.plot(pattern.numpy(), label='full pattern')
plt.legend()

mu_size = pattern[0].shape[0]

f = F(input_size=mu_size, hidden_size=50, output_size=mu_size)
losses = []

for epoch in range(50):
  # print("epoch {}".format(epoch))
  epoch_losses = []
  for offset in range(0, pattern.shape[0] - t_sample):
    loss = f.train_sample(pattern[offset:t_sample+offset], pattern[t_sample+offset])
    epoch_losses.append(loss)
    #print("loss = {}".format(loss))
    #print("------------")
  #pdb.set_trace()
  losses.append(np.mean(epoch_losses))

plt.figure(figsize=figsize)
plt.plot(losses, label='loss for f()')
plt.legend()

# %%
# A few sequential predictions
offset = 10
prediction_count = 50
predictions = f.predict(pattern[offset:t_sample+offset], prediction_count)

plt.figure(figsize=figsize)
plt.plot(range(0, t_sample + prediction_count + 1), pattern[offset:t_sample + offset + prediction_count + 1].numpy(), label='seed')
plt.plot(range(t_sample + 1, t_sample + prediction_count + 1), predictions, label='sequential predictions')
plt.legend()

# %%
# Expected precision
# Let's try training on expected error
f_precision = F(input_size=mu_size, hidden_size=50, output_size=1)
losses = []

for epoch in range(20):
  # print("epoch {}".format(epoch))
  epoch_losses = []
  for offset in range(0, pattern.shape[0] - t_sample):
    (output, hidden) = f.run_sample(pattern[offset:t_sample+offset])
    error = pattern[t_sample+offset] - output
    error = torch.abs(error.squeeze(0))
    #print("error = {}".format(error))
    loss = f_precision.train_sample(pattern[offset:t_sample+offset], error)
    epoch_losses.append(loss)
    #print("loss = {}".format(loss))
    #print("------------")
  #pdb.set_trace()
  losses.append(np.mean(epoch_losses))

plt.figure(figsize=figsize)
plt.plot(losses, label='loss for f_precision()')
plt.legend()

# %%

# actual vs predicted error
actual_errors = []
predicted_errors = []

for offset in range(0, pattern.shape[0] - t_sample):
  (output, _) = f.run_sample(pattern[offset:t_sample+offset])
  actual_error = np.abs(pattern[t_sample+offset][0] - output.item())
  actual_errors.append(actual_error)

  (predicted_error, _) = f_precision.run_sample(pattern[offset:t_sample+offset])
  predicted_errors.append(predicted_error.item())

plt.figure(figsize=figsize)
plt.plot(actual_errors, label='actual error')
plt.plot(predicted_errors, label='predicted error')
plt.legend()
# %%
# %%