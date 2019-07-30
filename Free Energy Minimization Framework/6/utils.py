import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class SlidingWindowBuffer(object):
  def __init__(self, item_count):
    self.item_count = item_count
    self.buffer = []

  def append_item(self, item):
    print("append_item {}".format(item))
    # return None while gathering initial items
    if len(self.buffer) < self.item_count - 1:
      self.buffer.append(item)
      return None

    # once enough items, convert to np.array
    elif len(self.buffer) == self.item_count - 1:
      self.buffer.append(item)
      self.buffer = np.array(self.buffer)

    else:
      self.buffer = np.roll(self.buffer, -1, axis=0)
      self.buffer[-1] = item

    return self.buffer

class SampleDataPointsGenerator(object):
  def __init__(self, count=1):
    self.index = 0
    self.count = count

  def __next__(self):
    self.index += 1
    if self.count == 1:
      return np.sin(self.index/10.0 + np.random.random_sample() * 0.0) * np.cos(self.index/25.0) + np.random.random_sample() * 0.0
    elif self.count == 2:
      return [
        np.cos(self.index/10.0 + np.random.random_sample() * 0.2) * np.sin(self.index/5.0),
        np.sin(self.index/10.0 + np.random.random_sample() * 0.2) * np.cos(self.index/20.0)
      ]

def plot_history(loss_history, title=None):
  loss_history = np.array(loss_history)
  fig = plt.figure(figsize=(8,6))
  fig.suptitle(title, fontsize=16)
  plt.plot(loss_history[:, 0],"--",label='loss_g')
  plt.plot(loss_history[:, 1],"--",label='loss_f')
  plt.plot(loss_history[:, 2],"-",label='mu_bar',linewidth=1,alpha = 0.3)
  plt.plot(loss_history[:, 3],"-",label='mu_hat',linewidth=1,alpha = 0.3)
  plt.plot(loss_history[:, 4],"-",label='mu', linewidth=2)
  plt.plot(loss_history[:, 5],"-",label='mu_pred',linewidth=2, alpha = 0.5)

  plt.legend()
  plt.show()

def plot_1d(data, title=None):
  fig = plt.figure(figsize=(8,6))
  fig.suptitle(title, fontsize=16)
  plt.plot(data)