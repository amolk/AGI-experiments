import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pdb

from utils import SlidingWindowBuffer

class FunctionApproximatorNetworkLSTM(nn.Module):
  """FunctionApproximatorNetwork is used to implement the g() and f() functions"""
  def __init__(self, input_size, output_size, hidden_size=50):
    super().__init__()
    self.lstm1 = nn.LSTMCell(input_size, hidden_size)
    self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
    self.linear = nn.Linear(hidden_size, output_size)

    batch_size = 1
    self.h_t = torch.zeros(batch_size, hidden_size, dtype=torch.double)
    self.c_t = torch.zeros(batch_size, hidden_size, dtype=torch.double)
    self.h_t2 = torch.zeros(batch_size, hidden_size, dtype=torch.double)
    self.c_t2 = torch.zeros(batch_size, hidden_size, dtype=torch.double)

  def forward(self, x):
    self.h_t = self.h_t.detach()
    self.c_t = self.c_t.detach()
    self.h_t2 = self.h_t2.detach()
    self.c_t2 = self.c_t2.detach()

    self.h_t, self.c_t = self.lstm1(x, (self.h_t, self.c_t))
    self.h_t2, self.c_t2 = self.lstm2(self.h_t, (self.h_t2, self.c_t2))
    print("self.h_t = {}".format(self.h_t))
    return self.linear(self.h_t2)

  def predict(self, x, steps):
    outputs = []
    output = x
    for i in range(steps):
      output = self.forward(output).detach()
      outputs += [output]
      print("output {}".format(output))
    #outputs = torch.stack(outputs, 1).squeeze(2)
    return outputs

class FunctionApproximatorNetworkRNN(nn.Module):
  """FunctionApproximatorNetwork is used to implement the g() and f() functions"""
  def __init__(self, input_size, output_size, hidden_size=5):
    super().__init__()
    self.rnn1 = nn.RNNCell(input_size, hidden_size)
    self.rnn2 = nn.RNNCell(hidden_size, hidden_size)
    self.linear = nn.Linear(hidden_size, output_size)

    batch_size = 1
    self.h_t = torch.zeros(batch_size, hidden_size, dtype=torch.double)
    # self.c_t = torch.zeros(batch_size, hidden_size, dtype=torch.double)
    self.h_t2 = torch.zeros(batch_size, hidden_size, dtype=torch.double)
    # self.c_t2 = torch.zeros(batch_size, hidden_size, dtype=torch.double)

  def forward(self, x):
    self.h_t = self.h_t.detach()
    # self.c_t = self.c_t.detach()
    self.h_t2 = self.h_t2.detach()
    # self.c_t2 = self.c_t2.detach()

    self.h_t = self.rnn1(x, self.h_t)
    print("self.h_t {}".format(self.h_t))
    self.h_t2 = self.rnn2(self.h_t, self.h_t2)
    print("self.h_t2 {}".format(self.h_t2))
    return self.linear(self.h_t2)

  def predict(self, x, steps):
    outputs = []
    output = x
    for i in range(steps):
      output = self.forward(output).detach()
      outputs += [output]
      print("output {}".format(output))
    #outputs = torch.stack(outputs, 1).squeeze(2)
    return outputs

class FunctionApproximatorNetwork(nn.Module):
  """FunctionApproximatorNetwork is used to implement the g() and f() functions"""
  def __init__(self, input_size, output_size, hidden_size=5):
    super().__init__()
    self.g1 = nn.Linear(np.prod(input_size), hidden_size, bias=True)
    self.g2 = nn.Linear(hidden_size, np.prod(output_size), bias=True)
    self.act = nn.Tanh()
    self.input_buffer = SlidingWindowBuffer(input_size)
    self.input_buffer.buffer = np.zeros(input_size)

  def forward(self, x):
    self.input_buffer.append_item(x.numpy())
    print("input {}".format(self.input_buffer.buffer))
    x = torch.from_numpy(self.input_buffer.buffer)
    x = F.relu(self.g1(x))
    x = self.g2(x)
    x = self.act(x)
    print("output {}".format(x))
    return x

  def predict(self, x, steps):
    outputs = []
    output = x
    for i in range(steps):
      output = self.forward(output).detach()
      outputs += [output]
      print("input {}".format(self.input_buffer.buffer))
      print("output {}".format(output))
    #outputs = torch.stack(outputs, 1).squeeze(2)
    return outputs
