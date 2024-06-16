# %%
!pip install -q --upgrade pip
!pip install -q opencv-python scipy datasets scikit-learn

# %%
import sys
path = "/Users/amolk/work/AGI/AGI-experiments/2024-EngramSOM"
if path not in sys.path:
	sys.path.append(path)

# %%
experiment_name = "01.02"

# %%
import torch
from patternmachine.signal_source.mnist_source import MNISTSource

data_source = MNISTSource(invert=False)
data_source.imshow(64, filename=f"output/{experiment_name}/train_images.png")

# %%
from torch import nn
import numpy as np
from patternmachine.utils import show_image_grid
from matplotlib import pyplot as plt
from tqdm import trange

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
		self.alpha = self.alpha * np.exp(-iteration / n_iter)
		# self.sigma = self.sigma_initial * np.exp(-iteration / (n_iter / np.log(self.sigma_initial)))

	def train_som(self, data_source, n_iter=100):
		errors = []
		errors.append([0, self.evaluate_som(data_source)])
		for it in range(n_iter):
			print("Epoch", it)
			self.update_learning_parameters(it, n_iter)
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

				if i%10000 == 0:
					errors.append([it*data_source.item_count+i, self.evaluate_som(data_source)])
					plt.scatter(*zip(*errors))
					plt.show()
					show_image_grid(self.weights.detach().view(-1, data_source.height, data_source.width))
					plt.show()

					# update sigma to reduce the neighborhood
					self.sigma = self.sigma * 0.8

		errors.append([it*data_source.item_count+i, self.evaluate_som(data_source)])
		return errors

	def evaluate_som(self, data_source):
		errors = []
		for i in range(data_source.test_images.shape[0]):
			x = torch.tensor(data_source.test_images[i], dtype=torch.float32).view(-1)
			x = x.unsqueeze(0)
			dists = self.forward(x)
			bmu_index = torch.argmin(dists).item()
			bmu = self.weights[bmu_index]
			error = torch.norm(x - bmu)
			errors.append(error.item())
		error = np.mean(errors)

		train_image_data = torch.tensor(data_source.train_images, dtype=torch.float32).view(-1, data_source.height*data_source.width)
		dists = torch.cdist(train_image_data, self.weights)
		bmu_indices = torch.argmin(dists, dim=0)
		neuron_labels = data_source.train_labels[bmu_indices]

		test_image_data = torch.tensor(data_source.test_images, dtype=torch.float32).view(-1, data_source.height*data_source.width)
		dists = torch.cdist(test_image_data, self.weights)
		bmu_indices = torch.argmin(dists, dim=1)
		predicted_labels = neuron_labels[bmu_indices]

		# get accuracy, precision, recall, f1-score by comparing actual labels with predicted labels
		from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
		accuracy = accuracy_score(data_source.test_labels, predicted_labels)
		precision = precision_score(data_source.test_labels, predicted_labels, average='weighted')
		recall = recall_score(data_source.test_labels, predicted_labels, average='weighted')
		f1 = f1_score(data_source.test_labels, predicted_labels, average='weighted')
		print("Accuracy:", accuracy)
		print("Precision:", precision)
		print("Recall:", recall)
		print("F1 Score:", f1)

		return error

def create_image_grid(images, gw, gh, iw, ih):
	# Reshape the flat images array to a grid shape
	images_grid = np.array(images).reshape(gh, gw, ih, iw, -1)

	# Transpose the array to move the grid dimensions next to each other
	grid_image = images_grid.transpose(0, 2, 1, 3, 4)

	# Reshape the array to the final image shape
	grid_image = grid_image.reshape(gh * ih, gw * iw, -1)

	return grid_image

som_shape = (20, 20)
som = SOM(som_shape[0], som_shape[1], data_source.height*data_source.width, sigma=0.9, alpha=0.1)
errors = som.train_som(data_source, 3)
# errors has the format [[iteration, error], ...]
# plot the error scatter plot
plt.scatter(*zip(*errors))
plt.savefig(f"output/{experiment_name}/loss.png")
plt.show()
show_image_grid(som.weights.detach().view(-1, data_source.height, data_source.width), filename=f"output/{experiment_name}/som_weights.png")
plt.show()

# %%
errors = som.train_som(data_source, 3)
# %%
