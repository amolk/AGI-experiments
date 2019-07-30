import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn

from unit import Unit

class UnitStack(object):
  def __init__(self, units):
    super().__init__()
    self.units = units
    self.unit_count = len(units)

  def step(self, mu_item, mu_awareness, train=True):
    # before step initialization
    [self.units[layer].before_step() for layer in range(self.unit_count)]

    # forward error propagation
    self.units[0].add_mu_item(mu_item)
    self.units[-1].mu = mu_awareness

    for layer in range(1, self.unit_count):
      # compute mu using previous layer's predictions
      # mu is part of the signal the previous layer could not predict
      self.units[layer].add_mu_item((self.units[layer - 1].mu - (self.units[layer - 1].mu_hat + self.units[layer - 1].mu_bar)).detach()[-1])
      self.units[layer - 1].set_mu_next(self.units[layer].mu)

    # backward flow of predictions
    [self.units[layer].compute_predictions() for layer in range(self.unit_count - 1, -1, -1)]

    # train
    if train:
      for layer in range(self.unit_count):
        self.units[layer].train()

    # return stats
    return [self.units[layer].history() for layer in range(self.unit_count)]