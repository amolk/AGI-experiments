
# %%
t_sample

# %%
%cd ~/work/free-energy-minimization-framework/
# from IPython.display import display_html
# def restartkernel() :
#   display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
# restartkernel()
%load_ext autoreload
%autoreload 2

# %%
import torch
from unit import Unit
from unit_stack import UnitStack
from utils import plot_history, SampleDataPointsGenerator, plot_1d
import pdb
from torchviz import make_dot
import numpy as np

# %%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_num_threads(12)

t_sample = 10
mu_size = t_sample
unit_count = 2
units = [Unit(name="unit{}".format(i), layer_index=i, mu_size=mu_size, mu_next_size=mu_size, device=device) for i in range(unit_count)]

network = UnitStack(units)

loss_history = []
mu_awareness = np.ones((t_sample,))

mu_awareness = torch.tensor(mu_awareness, requires_grad=True).float().to(device)
data_generator = SampleDataPointsGenerator()

# %%
# ----------- TRAIN -------------
for i in range(1000):
  loss = network.step(mu_item=next(data_generator), mu_awareness=mu_awareness, train=True)
  # print(loss)

  loss_history.append(loss)

  if (i+1) % 500 == 0:
    [plot_history(np.array(loss_history)[:, i, :], title='unit {}'.format(i)) for i in range(unit_count)]

    loss_history = []

print("==================")

# %%
# ----------- TEST -------------
next_mu_item = next(data_generator)
for i in range(20):
  loss = network.step(mu_item=next_mu_item, mu_awareness=mu_awareness, train=False)
  next_mu_item = loss[0][6][0].detach().numpy()
  # plot_1d(loss[0][6].detach().numpy(), title="mu")
  # plot_1d(loss[0][7].detach().numpy(), title="mu_bar")
  # plot_1d(loss[0][8].detach().numpy(), title="mu_hat")

  loss_history.append(loss)

  if (i+1) % 10 == 0:
    [plot_history(np.array(loss_history)[:, i, :], title='unit {}'.format(i)) for i in range(unit_count)]

    loss_history = []

print("==================")



# %%
