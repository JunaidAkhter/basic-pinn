from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

TOTAL_TIME = 12
a = 1.0
b = a
c = a
d = a

#Initial values of the prey and the predator
U0 = 1.0
V0 = 1.0


class U(nn.Module):
    """Simple neural network accepting one feature as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False):

        super().__init__()

        self.pinning = pinning

        self.layer_in = nn.Linear(1, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act
    def forward(self, x):
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)

class V(nn.Module):
    """Simple neural network accepting one features as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False):

        super().__init__()

        self.pinning = pinning

        self.layer_in = nn.Linear(1, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act
    def forward(self, x):
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)

def f(nn, x: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""

    return nn(x)

def df(nn, x: torch.Tensor = None, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = f(nn, x)
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value

def compute_loss(
    nns: [U, V], x: torch.Tensor = None, verbose: bool = False
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss

    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """
    nn1 = nns[0]
    nn2 = nns[1]  

    interior_loss_u = df(nn1, x) - a * f(nn1, x) + b * f(nn1, x) * f(nn2, x)

    interior_loss_v = df(nn2, x) - d * f(nn2, x) + b * f(nn2, x)

    interior_loss = interior_loss_u + interior_loss_v

    boundary = torch.Tensor([0.0])
    boundary.requires_grad = True
    boundary_loss_u = f(nn1, boundary) - U0
    boundary_loss_v = f(nn2, boundary) - V0
    final_loss = interior_loss.pow(2).mean() + boundary_loss_u ** 2 +\
        boundary_loss_u ** 2

    return final_loss




def train_model(
    nns: [U, V],
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
) -> [U, V]:

    loss_evolution = []

    optimizer = torch.optim.SGD(nns[0].parameters(), lr=learning_rate)
    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

            loss_evolution.append(loss.detach().numpy())

        except KeyboardInterrupt:
            break

    return nns, np.array(loss_evolution)



if __name__ == "__main__":
    from functools import partial

    domain = [0.0, 1.0]
    x = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True)
    x = x.reshape(x.shape[0], 1)

    nns = [U(4, 10), V(4, 10)]
    # f_initial = f(nn_approximator, x)
    # ax.plot(x.detach().numpy(), f_initial.detach().numpy(), label="Initial NN solution")

    # train the PINN
    loss_fn = partial(compute_loss, x=x, verbose=True)

    nn_approximator_trained, loss_evolution = train_model(
        nns, loss_fn=loss_fn, learning_rate=0.1, max_epochs=20_000
    )

    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
