import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class F(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.h2h = nn.Linear(hidden_size, hidden_size)

    self.i2o = nn.Linear(input_size + hidden_size, output_size)
    self.tanh = nn.Tanh()

    self.criterion = nn.MSELoss()
    self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

  def forward(self, input, hidden):
    combined = torch.cat((input, hidden), 1)

    hidden = self.i2h(combined)
    hidden = self.h2h(hidden)
    hidden = self.tanh(hidden)

    output = self.i2o(combined)
    output = self.tanh(output)

    return output, hidden

  def init_hidden(self, batch_size=1):
    return Variable(torch.zeros(batch_size, self.hidden_size)).float()

  def train_sample(self, sample, target):
    self.zero_grad()
    hidden = self.init_hidden()
    for input in sample:
      #print("hidden = {}".format(hidden))
      #print("input = {}".format(input))
      # pdb.set_trace()
      output, hidden = self.forward(input.unsqueeze(0), hidden)
      #print("output = {}".format(output))
      #print("hidden = {}".format(hidden))
      #print("--")

    target = target.unsqueeze(0)
    loss = self.criterion(output, target)
    loss.backward()
    self.optimizer.step()
    #print("output = {0}, target = {1}, loss = {2}".format(output.item(), target.item(), loss))
    #print("----")
    return loss.item()

  def train_batch(self, batch_samples, batch_targets):
    self.zero_grad()
    batch_size = batch_samples.shape[0]
    sample_size = batch_samples.shape[1]
    hidden = self.init_hidden(batch_size)
    outputs = []

    for i in range(sample_size):
      input = batch_samples[:, i]
      output, hidden = self.forward(input, hidden)
      outputs.append(output)

    outputs = torch.stack(outputs, 1)
    loss = self.criterion(outputs, batch_targets)
    loss.backward()

    self.optimizer.step()
    #print("output = {0}, target = {1}, loss = {2}".format(output.item(), target.item(), loss))
    #print("----")
    return loss.item()

  def run_sample(self, sample):
    with torch.no_grad():
      hidden = self.init_hidden()
      for input in sample:
        output, hidden = self.forward(input.unsqueeze(0), hidden)

      return output, hidden

  def predict(self, warmup_sample, prediction_count):
    with torch.no_grad():
      (output, hidden) = self.run_sample(warmup_sample)

      predictions = []
      for _ in range(prediction_count):
        output, hidden = self.forward(output, hidden)
        predictions.append(output)

      return predictions
