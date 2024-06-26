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
from patternmachine.similarity_utils import overlap_similarity


class DebugSettings:
	def __init__(self):
		self.debug_metrics = True
		self.debug_interval = 10000

class SOM(nn.Module):
	def __init__(self, som_height, som_width, pattern_height, pattern_width, alpha=0.5, beta=0.99, sigma=None, refractory_period=None):
		super(SOM, self).__init__()
		self.som_height = som_height
		self.som_width = som_width
		self.pattern_height = pattern_height
		self.pattern_width = pattern_width
		self.dim = pattern_height * pattern_width
		self.alpha = alpha
		self.beta = beta
		self.sigma_initial = sigma if sigma else 0.75
		self.sigma = self.sigma_initial
		self.refractory_period = refractory_period
		self.debug_settings = DebugSettings()

		self.weights = nn.Parameter(torch.rand(som_height * som_width, self.dim))
		self.precision = nn.Parameter(torch.rand(som_height * som_width, self.dim))

		# Create a grid of neuron positions
		self.grid = self.create_grid(som_height, som_width)

		# Initialize the last activation times with large negative values
		if self.refractory_period is not None:
			self.last_activation = -torch.ones(self.som_height * self.som_width, dtype=torch.int) * 99999

	def create_grid(self, som_height, som_width):
		return torch.tensor([[i, j] for i in range(som_height) for j in range(som_width)], dtype=torch.float32)

	def forward(self, x):
		# dists = torch.cdist(x, self.weights)
		# return dists
		sim = self.similarity(x, torch.ones_like(x), self.weights, self.precision)
		sim = sim.mean(-1)
		return 1.0-sim

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
					mask = (i - self.last_activation) < self.refractory_period
					dists[0, mask] = float('inf')

				bmu_index = torch.argmin(dists).item()
				bmu_loc = self.grid[bmu_index]

				# Update the last activation time for the BMU
				if self.refractory_period is not None:
					self.last_activation[bmu_index] = i

				# Compute the distance from the BMU to all other neurons
				dists_to_bmu = torch.norm(self.grid - bmu_loc, dim=1)
				influence = torch.exp(-dists_to_bmu / (2 * (self.sigma ** 2))).unsqueeze(1)

				# Update weights vectorized
				self.weights.data += self.alpha * influence * (x - self.weights)
				# precision beta decide how much to update pattern precision
				# assuming beta = 0.9
				# For winner
				# 	- influence is 1, so precision_beta is same as beta 0.9
				#		- as a result, precision for the winner will be updated (1-0.9), i.e. 10%
				# For far away from winner
				#		- influence is 0, so precision_beta is 0.9+(1-0)(1-0.9)=1.0
				#		- as a result, precision for this neuron will not be updated at all
				# For a neuron near the winner
				#		- say influence is 0.4, then precision_beta is 0.9+(1-0.4)(1-0.9)=0.96
				#		- as a result, precision for this neuron will be slightly updated (1-0.96), i.e 4%
				precision_beta = self.beta + (1 - influence) * (1 - self.beta)

				# precision is inverse of the difference between the (updated) patterns and the input pixels
				# if a pattern pixel is same as the input
				#		- difference = 0.0
				#		- precision = 1.0
				# if a pattern pixel is maximally different from the input (i.e. 0 vs 1)
				#		- precision = 1.0 - 1.0 = 0
				self.precision.data = (self.precision.data * precision_beta) + ((1-(x - self.weights).abs()) * (1 - precision_beta)) * x
				# self.precision[bmu_index].data = (self.precision[bmu_index].data * precision_beta) + ((1-(x - self.weights[bmu_index]).abs()) * (1 - precision_beta))

				if i%self.debug_settings.debug_interval == 0:
					self.log_metrics(metrics, data_source, it, i)

				if i%10000 == 0:
					# update sigma to reduce the neighborhood
					self.update_learning_parameters(it, n_iter)

		self.log_metrics(metrics, data_source, it, i)
		return metrics

	def similarity(self, x, x_precision, y, y_precision):
		# x = x.unsqueeze(1)
		# x_precision = x_precision.unsqueeze(1)

		# y = y.unsqueeze(0)
		# y_precision = y_precision.unsqueeze(0)

		# min_precision = 

		# sim = 1.0 / (1.0 + (x-y).abs())
		# sim = (1 - y_precision) * 0.2 + (sim * (y_precision + 1.0) * 0.5).pow(2)
		# sim = sim.clamp(min=0, max=1)
		# sim = sim.mean(dim=-1)
		# return sim
		sim = overlap_similarity(x, x_precision, y, y_precision) * x * x_precision
		# show_image_grid(sim.view(-1, data_source.height, data_source.width), vmax=None, vmin=None)

		return sim

	def log_metrics(self, all_metrics, data_source, epoch, example_index):
		time_step = self.get_time_step(data_source, epoch, example_index)
		metrics = self.evaluate_som(data_source)
		all_metrics[time_step] = metrics
		wandb.log({"time_step": time_step, **metrics})
		show_image_grid(self.weights.detach().view(-1, data_source.height, data_source.width), filename=f"output/{EXPERIMENT_NAME}/som_weights.png", wandb_name="som_weights")
		show_image_grid(self.precision.detach().view(-1, data_source.height, data_source.width), filename=f"output/{EXPERIMENT_NAME}/som_precision.png", wandb_name="som_precision")

	def get_time_step(self, data_source, epoch, example_index):
		return epoch * data_source.item_count + example_index

	def evaluate_som(self, data_source):
		metrics = {}
		# losses = []
		# for i in range(data_source.test_images.shape[0]):
		# 	x = torch.tensor(data_source.test_images[i], dtype=torch.float32).view(-1)
		# 	x = x.unsqueeze(0)
		# 	dists = 1-self.forward(x)
		# 	bmu_index = torch.argmin(dists).item()
		# 	bmu = self.weights[bmu_index]
		# 	loss = torch.norm(x - bmu)
		# 	losses.append(loss.item())
		# loss = np.mean(losses)
		# metrics = {"loss": loss}

		# train_image_data = torch.tensor(data_source.train_images, dtype=torch.float32).view(-1, data_source.height*data_source.width)
		# dists = 1-self.similarity(train_image_data, torch.ones_like(train_image_data), self.weights, self.precision)
		# bmu_indices = torch.argmin(dists, dim=0)
		# neuron_labels = data_source.train_labels[bmu_indices]

		# test_image_data = torch.tensor(data_source.test_images, dtype=torch.float32).view(-1, data_source.height*data_source.width)
		# dists = 1-self.similarity(test_image_data, torch.ones_like(test_image_data), self.weights, self.precision)
		# bmu_indices = torch.argmin(dists, dim=1)
		# predicted_labels = neuron_labels[bmu_indices]

		# # get accuracy, precision, recall, f1-score by comparing actual labels with predicted labels
		# accuracy = accuracy_score(data_source.test_labels, predicted_labels)
		# precision = precision_score(data_source.test_labels, predicted_labels, average='weighted')
		# recall = recall_score(data_source.test_labels, predicted_labels, average='weighted')
		# f1 = f1_score(data_source.test_labels, predicted_labels, average='weighted')

		# metrics = {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
		return metrics

# %%
wandb.init(project=PROJECT_NAME, name=f"{EXPERIMENT_NAME}_{wandb.util.generate_id()}", mode="disabled")

som_shape = (7, 7)
alpha = 0.2
sigma = 0.8
epochs = 3
som = SOM(*som_shape, data_source.height, data_source.width, sigma=sigma, alpha=alpha,
refractory_period=None, beta=0.99)

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
from patternmachine.similarity_utils import _overlap_similarity
# test overlap_similarity
x = torch.linspace(0.0, 1.0, 100).unsqueeze(0).expand(100, 100)
plt.imshow(x, cmap="gray")
plt.show()

y = x.clone().transpose(0, 1)
plt.imshow(y, cmap="gray")
plt.show()

plt.imshow(1-(x-y).abs(), cmap="gray")
plt.show()

p = 0.3

plt.imshow(overlap_similarity(x, torch.ones_like(x) * p, y, torch.ones_like(x) * p), cmap="gray")
plt.show()

# plt.imshow(_overlap_similarity(x, torch.ones_like(x) * p, y, torch.ones_like(x) * p), cmap="gray")
# plt.show()


# %%
