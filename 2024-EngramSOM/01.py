# %%
!python -m pip install --upgrade pip
!pip install opencv-python

# %%
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Extracting Frames from UCF101 Video
def extract_frames(video_path, frame_count=100, skip_frames=0):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    skipped = 0

    while cap.isOpened() and count < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (frame.shape[1]//5, frame.shape[0]//5))
        frames.append(frame.copy())
        count += 1

        # Skip the specified number of frames
        for _ in range(skip_frames):
            if not cap.read()[0]:
                break

    cap.release()
    return np.array(frames)

def preprocess_frames(frames):
    reshaped_frames = frames.reshape(frames.shape[0], -1).astype(np.float32)
    reshaped_frames /= 255.0
    return torch.tensor(reshaped_frames)

# Adjust this path to point to one of the UCF101 video files
# video_path = "/Users/amolk/Downloads/VID_20200411_175328.mp4" #"08_cat_small.m4v"
# frames = extract_frames(video_path)
# frame_shape = frames.shape[1:]
# frames_tensor = preprocess_frames(frames)

# %%


# %%

plt.figure(figsize=(1, 200))
plt.imshow(create_image_grid(frames_tensor, 1, frames.shape[0], frame_shape[1], frame_shape[0]), cmap='gray', vmin=0, vmax=1)

# %%

# Step 2: Implementing the SOM in PyTorch
class SOM(nn.Module):
    def __init__(self, m, n, dim, n_iter=100, alpha=0.3, sigma=None, refractory_period=3):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        self.sigma_initial = sigma if sigma else max(m, n) / 200.0
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

    def train_som(self, data):
        for it in range(self.n_iter):
            self.update_learning_parameters(it)
            for x in data:
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

som_shape = (10, 10)
som = SOM(som_shape[0], som_shape[1], frames_tensor.shape[1], n_iter=10)
som.train_som(frames_tensor)

# Example usage: To view the learned SOM weights
som_weights = som.weights.detach().numpy()
ig = create_image_grid(som_weights, som_shape[0], som_shape[1], frame_shape[1], frame_shape[0])
plt.figure(figsize=(ig.shape[0]//100, ig.shape[1]//100))
plt.imshow(ig, cmap='gray', vmin=0, vmax=1)


# %%
plt.imshow(create_image_grid(som_weights, som_shape[0], som_shape[1], frame_shape[1], frame_shape[0]), cmap='gray', vmin=0, vmax=1)
# %%
# %%
