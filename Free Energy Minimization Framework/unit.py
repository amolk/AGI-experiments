import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from function_approximator_network import FunctionApproximatorNetwork
from utils import SlidingWindowBuffer

class Unit(nn.Module):
  def __init__(self, name, layer_index, mu_size, mu_next_size, device):
    super().__init__()
    self.device = device
    self.name = name
    self.mu_size = mu_size
    self.mu_next_size = mu_next_size

    self.g = FunctionApproximatorNetwork(input_size=mu_next_size, output_size=mu_size)
    self.f = FunctionApproximatorNetwork(input_size=mu_size, output_size=mu_size)
    self.loss_g = 0
    self.loss_f = 0

    lr = 0.2
    self.optimizer_g = optim.SGD(self.g.parameters(), lr=lr, momentum=0)
    self.optimizer_f = optim.SGD(self.f.parameters(), lr=lr, momentum=0)

    # this unit's state
    self.previous_mu = torch.zeros(mu_size).to(device)
    self.mu = torch.zeros(mu_size).to(device)
    self.mu_bar = torch.zeros(mu_size).to(device)
    self.mu_hat = torch.zeros(mu_size).to(device)

    # next unit's mu
    self.mu_next = torch.zeros(mu_next_size).to(device)

    # buffer for collecting mu
    self.mu_buffer = SlidingWindowBuffer(mu_size)

#     def pool(self, buffer):
#         x = np.reshape(np.array([b.detach().numpy() for b in buffer]), (1, 1, self.t_sample * self.temporal_pooling_size))
#         x = torch.nn.functional.avg_pool1d(torch.tensor(x), kernel_size=self.temporal_pooling_size)
#         return x.reshape(self.t_sample)

  def add_mu_item(self, mu_item):
    x = self.mu_buffer.append_item(mu_item)
    if x is not None:
      self.mu = torch.tensor(self.mu_buffer.buffer).float()

  def set_mu_next(self, mu_next):
    self.mu_next = mu_next

  def before_step(self):
    pass

  def compute_predictions(self):
    self.mu_bar = self.f(self.previous_mu)
    self.mu_hat = self.g(self.mu_next)

  def train(self):
    self.g.train()
    self.optimizer_g.zero_grad()
    self.mu_hat = self.g(self.mu_next)
    self.loss_g = F.mse_loss(self.mu_hat, self.mu - self.mu_bar)
    self.loss_g.backward()
    self.optimizer_g.step()

    self.f.train()
    self.optimizer_f.zero_grad()
    self.mu_bar = self.f(self.previous_mu)
    self.loss_f = F.mse_loss(self.mu_bar, self.mu)
    self.loss_f.backward()
    self.optimizer_f.step()

    self.previous_mu = self.mu.detach()

  def history(self):
    return [self.loss_g, self.loss_f, self.mu_bar[-1], self.mu_hat[-1], self.mu[-1], self.mu_bar[-1] + self.mu_hat[-1], self.mu, self.mu_bar, self.mu_hat]

