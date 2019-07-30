# %%
%cd ~/work/free-energy-minimization-framework/8/
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
t_sample = 30
learning_rate = 0.001
# sin wave with noise present only certain ranges
pattern_length = 400
pattern1 = torch.tensor(np.sin(np.arange(pattern_length/4) * 0.30) - 0.5 + np.random.sample(int(pattern_length/4)) * 0.0).float().unsqueeze(1)
pattern2 = torch.tensor(np.sin(np.arange(pattern_length/4) * 0.50) + np.random.sample(int(pattern_length/4)) * 0.0).float().unsqueeze(1)
pattern3 = torch.tensor(np.sin(np.arange(pattern_length/4) * 0.10) + 0.5 + np.random.sample(int(pattern_length/4)) * 0.3).float().unsqueeze(1)
pattern4 = torch.tensor(np.sin(np.arange(pattern_length/4) * 0.80) + np.random.sample(int(pattern_length/4)) * 0.0).float().unsqueeze(1)
pattern = torch.cat( (pattern1, pattern2, pattern3, pattern4) ) * 0.5
plt.figure(figsize=figsize)
plt.plot(pattern.numpy(), label='full pattern')
plt.legend()

# %%
mu_size = pattern[0].shape[0]

f = F(input_size=mu_size, hidden_size=200, output_size=mu_size, learning_rate=0.3)
losses = []

samples = []
targets = []
for offset in range(0, pattern.shape[0] - t_sample):
  samples.append(pattern[offset:t_sample+offset])
  targets.append(pattern[offset+1:t_sample+offset+1])

samples = torch.stack(samples)
targets = torch.stack(targets)

for epoch in range(1000):
  # print("epoch {}".format(epoch))
  loss = f.train_batch(samples, targets)
  losses.append(loss)

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
f_precision = F(input_size=mu_size, hidden_size=200, output_size=1, learning_rate=0.1)
losses = []
pattern_errors = []

for i in range(0, pattern.shape[0] - t_sample):
  (output, hidden) = f.run_sample(pattern[i:t_sample+i])
  # error = pattern[t_sample+i] - output
  error = (pattern[t_sample+i] - output) ** 2
  pattern_errors.append([error])
pattern_errors = torch.tensor(pattern_errors)

samples = []
targets = []
# first item in pattern_errors is for pattern[0:t_sample]
# we need t_sample length sample of pattern and pattern_error
# first t_sample length sample for pattern_error starts at offset t_sample on pattern
for offset in range(t_sample, pattern.shape[0] - t_sample * 2):
  samples.append(pattern[offset : t_sample + offset])
  targets.append(pattern_errors[offset - t_sample : offset])

samples = torch.stack(samples)
targets = torch.stack(targets)

for epoch in range(1000):
  # print("epoch {}".format(epoch))
  loss = f_precision.train_batch(samples, targets)
  losses.append(loss)

plt.figure(figsize=figsize)
plt.plot(losses, label='loss for f_precision()')
plt.legend()

# %%

# actual vs predicted error
actual_errors = []
predicted_errors = []

for offset in range(t_sample, pattern.shape[0] - t_sample * 2):
  actual_error = pattern_errors[offset]
  actual_errors.append(actual_error)

  (predicted_error, _) = f_precision.run_sample(pattern[offset:t_sample+offset])
  predicted_errors.append(predicted_error.item())

plt.figure(figsize=(18,12))
plt.title("Detecting context switch (near 80, 180, 280)")
plt.plot(actual_errors, '--', label='|actual error|', linewidth = 1.0)
plt.plot(predicted_errors, '--', label='predicted error', linewidth = 1.0)
plt.plot(np.array(actual_errors) - np.array(predicted_errors), label='|actual-predicted error|', linewidth = 2.0)
plt.legend()
# %%

# %%