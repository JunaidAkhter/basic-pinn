#%%
from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

#%%
LENGTH = 1.
TOTAL_TIME = 1.
n = 1 #frequency of sinusoidal initial conditions

def diff_eq_exact_solution(x, t):
    """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    return np.exp(-t) * np.sin(n * np.pi * x / LENGTH)


def gen_exact_solution(n_points_x, n_points_t):
    """Generates exact solution for the heat equation for the given values of x and t."""
    # Number of points in each dimension:
    x_dim, t_dim = (n_points_x, n_points_t)

    # Bounds of 'x' and 't':
    x_min, t_min = (-LENGTH, 0.0)
    x_max, t_max = (LENGTH, TOTAL_TIME)

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    # Obtain the value of the exact solution for each generated point:
    for i in range(x_dim):
        for j in range(t_dim):
            usol[i][j] = diff_eq_exact_solution(x[i], t[j])

    # Save solution:
    
    return x, t, usol

x, t, usol = gen_exact_solution(100, 100)    
np.savez("diff_eq_data_ext", x=x, t=t, usol=usol)
#%%
location = "/home/junaid/Downloads/Paderborn/PhD/Research_directions/MultiCriteria/phd_code/basic-pinn/diffusion_equation"
data = np.load("diff_eq_data_ext.npz")
x, t, usol = data["x"], data["t"], data["usol"]
#%%
print(x.shape, t.shape, usol.shape)

plt.plot(x, usol[:, 0], label = "t = 0")
plt.plot(x, usol[:, -1], label = "t = 1")
plt.legend()
plt.show()