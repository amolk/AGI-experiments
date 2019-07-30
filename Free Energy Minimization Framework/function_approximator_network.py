import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class FunctionApproximatorNetwork(nn.Module):
  """FunctionApproximatorNetwork is used to implement the g() and f() functions"""
  def __init__(self, input_size, output_size, hidden_size=10):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.g1 = nn.Linear(np.prod(input_size), hidden_size, bias=True)
    self.g2 = nn.Linear(hidden_size, np.prod(output_size), bias=True)
    self.act = nn.Tanh()

  def forward(self, x):
    x = x.flatten()
    x = F.relu(self.g1(x))
    x = self.g2(x)
    x = self.act(x)
    return x
