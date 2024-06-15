# %%
!pip install -q --upgrade pip
!pip install -q opencv-python scipy

# %%
import sys
path = "/Users/amolk/work/AGI/AGI-experiments/2024-EngramSOM"
if path not in sys.path:
	sys.path.append(path)

# %%
experiment_name = "01.01"

# %%
import torch
from patternmachine.signal_source.video_source import VideoSource
from matplotlib import pyplot as plt

video_path = "../assets/eli_walk.avi"
video_source = VideoSource(filepath=video_path, stride=1, height=200, width=200, invert=True)
video_source.imshow(64, filename=f"output/{experiment_name}/train_images.png")

# %%
from torch import nn
import numpy as np
from patternmachine.utils import show_image_grid
from matplotlib import pyplot as plt

class SOM(nn.Module):
	def __init__(self, m, n, dim, n_iter=100, alpha=0.5, sigma=None, refractory_period=3):
		super(SOM, self).__init__()
		self.m = m
		self.n = n
		self.dim = dim
		self.n_iter = n_iter
		self.alpha = alpha
		self.sigma_initial = sigma if sigma else max(m, n) / 100.0
		self.sigma = self.sigma_initial
		self.refractory_period = refractory_period

		self.weights = nn.Parameter(torch.rand(m * n, dim))

		# Create a grid of neuron positions
		self.grid = self.create_grid(m, n)

		# Initialize the last activation times with large negative values
		self.last_activation = -torch.ones(m * n, dtype=torch.int)

	def create_grid(self, m, n):
		return torch.tensor([[i, j] for i in range(m) for j in range(n)], dtype=torch.float32)

	def forward(self, x):
		dists = torch.cdist(x, self.weights)
		return dists

	def update_learning_parameters(self, iteration):
		self.alpha = self.alpha * np.exp(-iteration / self.n_iter)
		self.sigma = self.sigma_initial * np.exp(-iteration / (self.n_iter / np.log(self.sigma_initial)))

	def train_som(self, video_source):
		for it in range(self.n_iter):
			self.update_learning_parameters(it)
			video_source.seek(0)
			images = video_source.item()
			for _ in range(video_source.item_count):
				x = next(images).view(-1)
				x = x.unsqueeze(0)
				dists = self.forward(x)

				# Mask out neurons that are in the refractory period
				mask = (it - self.last_activation) < self.refractory_period
				dists[0, mask] = float('inf')

				bmu_index = torch.argmin(dists).item()
				bmu_loc = self.grid[bmu_index]

				# Update the last activation time for the BMU
				self.last_activation[bmu_index] = it

				# Compute the distance from the BMU to all other neurons
				dists_to_bmu = torch.norm(self.grid - bmu_loc, dim=1)
				influence = torch.exp(-dists_to_bmu / (2 * (self.sigma ** 2))).unsqueeze(1)

				# Update weights vectorized
				self.weights.data += self.alpha * influence * (x - self.weights)

def create_image_grid(images, gw, gh, iw, ih):
	# Reshape the flat images array to a grid shape
	images_grid = np.array(images).reshape(gh, gw, ih, iw, -1)

	# Transpose the array to move the grid dimensions next to each other
	grid_image = images_grid.transpose(0, 2, 1, 3, 4)

	# Reshape the array to the final image shape
	grid_image = grid_image.reshape(gh * ih, gw * iw, -1)

	return grid_image

som_shape = (20, 20)
som = SOM(som_shape[0], som_shape[1], video_source.height*video_source.width, n_iter=100, sigma=0.75, alpha=1.0)
som.train_som(video_source)
show_image_grid(som.weights.detach().view(-1, video_source.height, video_source.width), filename=f"output/{experiment_name}/som_weights.png")

# %%
