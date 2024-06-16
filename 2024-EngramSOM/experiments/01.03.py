# %%
!pip install -q --upgrade pip
!pip install -q opencv-python scipy datasets scikit-learn wandb

# %%
import sys
import pathlib
path = pathlib.Path().absolute().parent
PROJECT_NAME = path.name
if str(path) not in sys.path:
	sys.path.append(str(path))

# %%
# Detect experiment name from the file name
EXPERIMENT_NAME = __file__.split("/")[-1].replace(".py", "") # e.g. "01.03"
print("Experiment:", EXPERIMENT_NAME)

# Create the output directory if it does not exist
import os
os.makedirs(f"output/{EXPERIMENT_NAME}", exist_ok=True)

# %%
import torch
from patternmachine.signal_source.mnist_source import MNISTSource

data_source = MNISTSource(invert=False)
data_source.imshow(64, filename=f"output/{EXPERIMENT_NAME}/train_images.png")

# %%
from torch import nn
import numpy as np
from patternmachine.utils import show_image_grid
from matplotlib import pyplot as plt
from tqdm import trange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb


class DebugSettings:
	def __init__(self):
		self.debug_metrics = True
		self.debug_interval = 10000

class SOM(nn.Module):
	def __init__(self, m, n, dim, alpha=0.5, sigma=None, refractory_period=None):
		super(SOM, self).__init__()
		self.m = m
		self.n = n
		self.dim = dim
		self.alpha = alpha
		self.sigma_initial = sigma if sigma else 0.75
		self.sigma = self.sigma_initial
		self.refractory_period = refractory_period
		self.debug_settings = DebugSettings()

		self.weights = nn.Parameter(torch.rand(m * n, dim))

		# Create a grid of neuron positions
		self.grid = self.create_grid(m, n)

		# Initialize the last activation times with large negative values
		if self.refractory_period is not None:
			self.last_activation = -torch.ones(m * n, dtype=torch.int)

	def create_grid(self, m, n):
		return torch.tensor([[i, j] for i in range(m) for j in range(n)], dtype=torch.float32)

	def forward(self, x):
		dists = torch.cdist(x, self.weights)
		return dists

	def update_learning_parameters(self, iteration, n_iter):
		# self.alpha = self.alpha * np.exp(-iteration / n_iter)
		self.sigma = self.sigma * 0.9
		wandb.log({"alpha": self.alpha, "sigma": self.sigma})

	def train_som(self, data_source, n_iter=100):
		metrics = {}
		self.log_metrics(metrics, data_source, 0, 0)
		for it in range(n_iter):
			print("Epoch", it)
			# data_source.seek(0)
			images = data_source.item()
			for i in trange(data_source.item_count):
				x = next(images).view(-1)
				x = x.unsqueeze(0)
				dists = self.forward(x)

				# Mask out neurons that are in the refractory period
				if self.refractory_period is not None:
					mask = (it - self.last_activation) < self.refractory_period
					dists[0, mask] = float('inf')

				bmu_index = torch.argmin(dists).item()
				bmu_loc = self.grid[bmu_index]

				# Update the last activation time for the BMU
				if self.refractory_period is not None:
					self.last_activation[bmu_index] = it

				# Compute the distance from the BMU to all other neurons
				dists_to_bmu = torch.norm(self.grid - bmu_loc, dim=1)
				influence = torch.exp(-dists_to_bmu / (2 * (self.sigma ** 2))).unsqueeze(1)

				# Update weights vectorized
				self.weights.data += self.alpha * influence * (x - self.weights)

				if i%self.debug_settings.debug_interval == 0:
					self.log_metrics(metrics, data_source, it, i)

				if i%10000 == 0:
					# update sigma to reduce the neighborhood
					self.update_learning_parameters(it, n_iter)

		self.log_metrics(metrics, data_source, it, i)
		return metrics

	def log_metrics(self, all_metrics, data_source, epoch, example_index):
		time_step = self.get_time_step(data_source, epoch, example_index)
		metrics = self.evaluate_som(data_source)
		all_metrics[time_step] = metrics
		wandb.log({"time_step": time_step, **metrics})
		show_image_grid(self.weights.detach().view(-1, data_source.height, data_source.width), filename=f"output/{EXPERIMENT_NAME}/som_weights.png", wandb_name="som_weights")

	def get_time_step(self, data_source, epoch, example_index):
		return epoch * data_source.item_count + example_index

	def evaluate_som(self, data_source):
		losses = []
		for i in range(data_source.test_images.shape[0]):
			x = torch.tensor(data_source.test_images[i], dtype=torch.float32).view(-1)
			x = x.unsqueeze(0)
			dists = self.forward(x)
			bmu_index = torch.argmin(dists).item()
			bmu = self.weights[bmu_index]
			loss = torch.norm(x - bmu)
			losses.append(loss.item())
		loss = np.mean(losses)

		train_image_data = torch.tensor(data_source.train_images, dtype=torch.float32).view(-1, data_source.height*data_source.width)
		dists = torch.cdist(train_image_data, self.weights)
		bmu_indices = torch.argmin(dists, dim=0)
		neuron_labels = data_source.train_labels[bmu_indices]

		test_image_data = torch.tensor(data_source.test_images, dtype=torch.float32).view(-1, data_source.height*data_source.width)
		dists = torch.cdist(test_image_data, self.weights)
		bmu_indices = torch.argmin(dists, dim=1)
		predicted_labels = neuron_labels[bmu_indices]

		# get accuracy, precision, recall, f1-score by comparing actual labels with predicted labels
		accuracy = accuracy_score(data_source.test_labels, predicted_labels)
		precision = precision_score(data_source.test_labels, predicted_labels, average='weighted')
		recall = recall_score(data_source.test_labels, predicted_labels, average='weighted')
		f1 = f1_score(data_source.test_labels, predicted_labels, average='weighted')

		metrics = {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
		return metrics

# %%
wandb.init(project=PROJECT_NAME, name=f"{EXPERIMENT_NAME}_{wandb.util.generate_id()}")

som_shape = (20, 20)
alpha = 0.1
sigma = 0.9
epochs = 3
som = SOM(som_shape[0], som_shape[1], data_source.height*data_source.width, sigma=sigma, alpha=alpha)

wandb.config.update({
	"som_height": som_shape[0],
	"som_width": som_shape[1],
	"pattern_height": data_source.height,
	"pattern_width": data_source.width,
	"alpha": alpha,
	"sigma": sigma,
	"epochs": epochs
})

metrics = som.train_som(data_source, epochs)

wandb.finish()
# %%
