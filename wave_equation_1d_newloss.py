from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


LENGTH = 1.
TOTAL_TIME = 1.


def initial_condition(x) -> torch.Tensor:
    res = torch.sin( 2*np.pi * x).reshape(-1, 1) * 0.5
    return res


class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh(), pinning: bool = False):

        super().__init__()

        self.pinning = pinning

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, t):

        x_stack = torch.cat([x, t], dim=1)        
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        # if requested pin the boundary conditions 
        # using a surrogate model: (x - 0) * (x - L) * NN(x)
        if self.pinning:
            logits *= (x - x[0]) * (x - x[-1])
        
        return logits

def f(nn_approximator: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    boundary_t = torch.zeros_like(t)
    return initial_condition(x) + (1 - torch.exp(-(t - boundary_t))) * nn_approximator(x, t)


def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value


def dfdt(nn_approximator: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the time variable of arbitrary order"""
    f_value = f(nn_approximator, x, t)
    #boundary_t = torch.zeros_like(t)
    #return (1 - torch.exp(-(t - boundary_t)))*df(f_value, t, order=order)

    return df(f_value, t, order=order)


def dfdx(nn_approximator: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(nn_approximator, x, t)
    return df(f_value, x, order=order)


def compute_loss(
    nn_approximator: PINN, x: torch.Tensor = None, t: torch.Tensor = None
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss

    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    C = 1.

    # PDE residual
    interior_loss = dfdx(nn_approximator, x, t, order=2) - (1/C**2) * dfdt(nn_approximator, x, t, order=2)

    # initial condition loss on both the function and its
    # time first-order derivative
    x_raw = torch.unique(x).reshape(-1, 1).detach().numpy()
    x_raw = torch.Tensor(x_raw)
    #x_raw.requires_grad = True

    #print("x raw inside shape:", x_raw.shape)
    #print(" t raw inside shape: ", t_raw.shape)

      
    t_initial = torch.zeros_like(x_raw)
    t_initial.requires_grad = True
    initial_loss_df = dfdt(nn_approximator, x_raw, t_initial, order=1)
    
    # obtain the final MSE loss by averaging each loss term and summing them up
    final_loss = \
        interior_loss.pow(2).mean() +\
        initial_loss_df.pow(2).mean()
 
    return final_loss


def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
) -> PINN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator


def check_gradient(nn_approximator: PINN, x: torch.Tensor, t: torch.Tensor) -> bool:

    eps = 1e-4
    
    dfdx_fd = (f(nn_approximator, x + eps, t) - f(nn_approximator, x - eps, t)) / (2 * eps)
    dfdx_autodiff = dfdx(nn_approximator, x, t, order=1)
    is_matching_x = torch.allclose(dfdx_fd.T, dfdx_autodiff.T, atol=1e-2, rtol=1e-2)

    dfdt_fd = (f(nn_approximator, x, t + eps) - f(nn_approximator, x, t - eps)) / (2 * eps)
    dfdt_autodiff = dfdt(nn_approximator, x, t, order=1)
    is_matching_t = torch.allclose(dfdt_fd.T, dfdt_autodiff.T, atol=1e-2, rtol=1e-2)
    
    eps = 1e-2

    d2fdx2_fd = (f(nn_approximator, x + eps, t) - 2 * f(nn_approximator, x, t) + f(nn_approximator, x - eps, t)) / (eps ** 2)
    d2fdx2_autodiff = dfdx(nn_approximator, x, t, order=2)
    is_matching_x2 = torch.allclose(d2fdx2_fd.T, d2fdx2_autodiff.T, atol=1e-2, rtol=1e-2)

    d2fdt2_fd = (f(nn_approximator, x, t + eps) - 2 * f(nn_approximator, x, t) + f(nn_approximator, x, t - eps)) / (eps ** 2)
    d2fdt2_autodiff = dfdt(nn_approximator, x, t, order=2)
    is_matching_t2 = torch.allclose(d2fdt2_fd.T, d2fdt2_autodiff.T, atol=1e-2, rtol=1e-2)
    
    return is_matching_x and is_matching_t and is_matching_x2 and is_matching_t2


def plot_solution(nn_trained: PINN, x: torch.Tensor, t: torch.Tensor):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    x_raw = torch.unique(x).reshape(-1, 1)
    t_raw = torch.unique(t)
        
    def animate(i):

        if not i % 10 == 0:
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            f_final = f(nn_trained, x_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy(), f_final.detach().numpy(), label=f"Time {float(t[i])}"
            )
            ax.legend()

    n_frames = t_raw.shape[0]
    ani = FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=False)
    writergif = matplotlib.animation.PillowWriter(fps=1)
    ani.save('wave_newloss.gif',writer=writergif)
    #ani.save("wave_newloss.mp4")
    plt.show()

if __name__ == "__main__":
    from functools import partial

    N = 150

    x_domain = [0.0, LENGTH]; n_points_x = N
    t_domain = [0.0, TOTAL_TIME]; n_points_t = N
    

    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x, requires_grad=True)
    t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points_t, requires_grad=True)
    grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
    

    #print("x_raw", x_raw)
    #print(" x_raw shape", x_raw.shape)
    #print("t_raw", t_raw)
    #print("t_raw shape", t_raw.shape)
    #print("grids", grids)
    #print("shape of grids 0", grids[0].shape, "shape of grids 1", grids[1].shape)



    x = grids[0].flatten().reshape(-1, 1)
    t = grids[1].flatten().reshape(-1, 1)


    #print("x grids shape:", x.shape)
    #print("t grids shape:", t.shape)

    #print("value of x after flattening:", x)

    # plt.figure(1)
    # plt.plot(x_raw.detach().numpy(), initial_condition(x_raw).detach().numpy())
    # plt.show()

    nn_approximator = PINN(3, 15, pinning=False)
    # assert check_gradient(nn_approximator, x, t)
    
    compute_loss(nn_approximator, x=x, t=t)
    
    # train the PINN
    loss_fn = partial(compute_loss, x=x, t=t)
    nn_approximator_trained = train_model(
        nn_approximator, loss_fn=loss_fn, learning_rate=0.005, max_epochs=20_000
    )

    input("Plot")
    plot_solution(nn_approximator_trained, x, t)
