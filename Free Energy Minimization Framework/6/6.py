# %%
%cd ~/work/free-energy-minimization-framework/6/
# from IPython.display import display_html
# def restartkernel() :
#   display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
# restartkernel()
%load_ext autoreload
%autoreload 2

# %%
import torch
import torch.nn as nn
import torch.optim as optim
# from unit import Unit
# from unit_stack import UnitStack
from utils import plot_history, SampleDataPointsGenerator, plot_1d
from function_approximator_network import FunctionApproximatorNetwork
import pdb
from torchviz import make_dot
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# set random seed to 0
np.random.seed(1)
torch.manual_seed(1)

# build the model
f = FunctionApproximatorNetwork(input_size=10, output_size=1)
f.double()
criterion = nn.MSELoss()
optimizer = optim.SGD(f.parameters(), lr=0.8)
data = SampleDataPointsGenerator()

input_history = []
input = torch.tensor([next(data)]).double()
input_history.append(input)
for i in tqdm(range(300)):
  optimizer.zero_grad()
  f.train()
  out = f(input)

  target = torch.tensor([next(data)]).double()
  loss = criterion(out, target[0])

  loss.backward()
  input = target
  input_history.append(input)
  optimizer.step()

input_history = input_history[-100:]
plt.plot(np.arange(len(input_history)), input_history)

pred = f.predict(torch.tensor([next(data)]).double(), 100)
plt.plot(np.arange(len(input_history), len(input_history) + len(pred)), pred)


#   # begin to predict, no need to track gradient here
#   with torch.no_grad():
#     future = 1000
#     pred = seq(test_input, future=future)
#     loss = criterion(pred[:, :-future], test_target)
#     print('test loss:', loss.item())
#     y = pred.detach().numpy()
#   # draw the result
#   plt.figure(figsize=(30,10))
#   plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
#   plt.xlabel('x', fontsize=20)
#   plt.ylabel('y', fontsize=20)
#   plt.xticks(fontsize=20)
#   plt.yticks(fontsize=20)
#   def draw(yi, color):
#     plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
#     plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
#   draw(y[0], 'r')
#   draw(y[1], 'g')
#   draw(y[2], 'b')
#   #plt.savefig('predict%d.pdf'%i)
#   plt.show()

# #%%
#     # draw the result
#     plt.figure(figsize=(30,10))
#     plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
#     plt.xlabel('x', fontsize=20)
#     plt.ylabel('y', fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     def draw(yi, color):
#         plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
#         plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
#     draw(y[0], 'r')
#     draw(y[1], 'g')
#     draw(y[2], 'b')
#     #plt.savefig('predict%d.pdf'%i)
#     plt.show()

# %%
pred

# %%