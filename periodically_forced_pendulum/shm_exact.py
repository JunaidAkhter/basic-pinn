#%%
from typing import Callable
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from torch import nn
import numpy as np
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#%%

# defining the parameters
k = 1.0
m = 1.0
ω = np.sqrt(k/m)

A = 2*m/ω
B = 1*m 

def exact_solution(A, B, x):

    return A*torch.sin(ω*x) + B*torch.cos(ω*x)



domain = [0.0, 10.0]
x = torch.linspace(domain[0], domain[1], steps=200)

u_sol = exact_solution(A, B, x)

fig, ax = plt.subplots()

ax.scatter(x, u_sol, color="red")
ax.plot(x, u_sol, label="Exact solution")
"""   """
ax.set(title="Simple Harmonic Oscillator", xlabel="t", ylabel="f(t)")
ax.legend()
plt.show()
