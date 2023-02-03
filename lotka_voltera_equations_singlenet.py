#%%
# from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from scipy.integrate import solve_ivp

#%%
class PINN(nn.Module):
    """Simple neural network accepting one feature as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.ReLU(), pinning: bool = False):
        # MB what does `pinning` doe?
        super().__init__()

        self.pinning = pinning

        self.layers = nn.Sequential()

        self.layers.add_module("layer_in", nn.Linear(1, dim_hidden))

        for i in range(num_hidden):
            self.layers.add_module(f"hidden_ReLU{i}", act)
            self.layers.add_module(f"hidden_layer{i}", nn.Linear(dim_hidden, dim_hidden))
        
        self.layers.add_module("final_ReLu", act)
        self.layers.add_module("layer_out", nn.Linear(dim_hidden, 2))

    def forward(self, x):
        return self.layers(x)

    def grads(self, vals_out, vals_in):
        if vals_out.dim() == 1:
            #vals_in = vals_in.reshape(1, -1) # turn into single-row matrix
            vals_out = vals_out.reshape(1, - 1)
        
        grads = []
        for output_index in range(vals_out.shape[1]):
            out_i = vals_out[:, output_index] # values of output at different points in time
            out_dt = torch.autograd.grad(
                out_i,
                vals_in,
                grad_outputs=torch.ones_like(out_i),
                create_graph=True,
                retain_graph=True,
            )[0].reshape(-1)
            grads.append(out_dt)
        return grads
#%%

def loss(nna, t, t0 = torch.Tensor([0.0]), states0 = torch.Tensor([U0, V0])):
    states = nna(t)
    grads = nna.grads(states, t)

    u = states[:, 0]
    v = states[:, 1]

    dudt = grads[0]
    dvdt = grads[1]

    # interior loss:
    # du/dt = α u - β u v
    # dv/dt = δ u v - γ v

    # f = [ au - βuv - du/dt, δuv - γv - dv/dt ]    
       
    f0 = a * u - b * u * v - dudt
    f1 = c * u * v - d * v - dvdt
    
    # Σ_i (||f_{i-1}|| + ||f_i||)/2 Δt_i
    Δt = (t[1:] - t[:-1]).reshape(-1)
    F = torch.sqrt(f0**2 + f1**2)

    loss_interior = torch.sum( Δt * (F[:-1] + F[1:])/2 )

    # boundary loss
    pred0 = nna(t0)
    loss_boundary = torch.sqrt(torch.sum((pred0 - states0)**2))

    return loss_interior + loss_boundary, loss_interior, loss_boundary

#%%
def train_model(
    nn_approximator:PINN,
    loss_fn,
    learning_rate = 0.001,
    max_epochs: int = 500,
) -> PINN:

    loss_evolution = []

    everynth = int(np.ceil(max_epochs/10))
    optimizer = torch.optim.SGD(nn_approximator.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):
        try:
            optimizer.zero_grad()
            loss, li, lb = loss_fn(nn_approximator)
            loss.backward()
            optimizer.step()

            if epoch % everynth == 0:
                print(f"Epoch: {epoch} - Loss: {float(loss):>7f} = {li.item():>3f} + {lb.item():>3f}")

            loss_evolution.append(loss.item())

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_evolution)
#%%

TOTAL_TIME = 2.0
a = 1.0
b = a
c = a
d = a

U0 = 1.5
V0 = 1.0
        
domain = [0.0, TOTAL_TIME]
t = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True).reshape(-1,1)
nna = PINN(2, 50)

model_loss = lambda nna : loss(nna, t, t0=torch.Tensor([0.0,]), states0 = torch.Tensor([U0, V0]))
nna, loss_evolution = train_model(nna, model_loss, 0.001, 5000);

#%%
T = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)

fig, ax = plt.subplots()

f_final_training = nna(t).detach().numpy()
f_final = nna(T).detach().numpy()

_t = t.detach().numpy().reshape(-1)
_T = T.detach().numpy().reshape(-1)

ax.scatter(_t, f_final_training[:, 0], label="Training points u", color="red")
ax.scatter(_t, f_final_training[:, 1], label="Training points v", color="black")

ax.plot(_T, f_final[:, 0], label="NN final solution u", color = "red")
ax.plot(_T, f_final[:, 1], label="NN final solution v", color = "black")

ax.set(title="Lotka_voltera_equations", xlabel="t", ylabel="Population")

def lveqs(t, states):
    u = states[0]
    v = states[1]
    return np.array([
        a * u - b * u * v, 
        c * u * v - d * v
    ])
    # du/dt = α u - β u v
    # dv/dt = δ u v - γ v

numeric_solution = solve_ivp(lveqs, (_T[0], _T[-1]), [U0, V0], "RK45", _T)

usol = numeric_solution.y[0, :]
vsol = numeric_solution.y[1, :]

ax.plot(numeric_solution.t, usol)
ax.plot(numeric_solution.t, vsol)

ax.legend()

fig, ax = plt.subplots()
ax.semilogy(loss_evolution)
ax.set(title="Loss evolution", xlabel="# epochs", ylabel="Loss")
ax.legend()

plt.show()
# %%
