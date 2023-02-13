#%%
from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

#%%
LENGTH = 17.5
TOTAL_TIME = 20
n = 2 #frequency of sinusoidal initial conditions

def initial_condition(x) -> torch.Tensor:
    res = torch.sin(n * np.pi * x/LENGTH).reshape(-1, 1)
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
    return nn_approximator(x, t)


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
    return df(f_value, t, order=order)


def dfdx(nn_approximator: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(nn_approximator, x, t)
    return df(f_value, x, order=order)

#%%
def compute_loss(
    nn_approximator: PINN, x: torch.Tensor = None, t: torch.Tensor = None
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """
    #thernal diffusivity constant
    alpha = 0.4

    # PDE residual
    interior_loss = dfdx(nn_approximator, x, t, order=2) - (1/alpha) * dfdt(nn_approximator, x, t, order=1)


    interior_loss = dfdt(nn_approximator, x, t, order=1) + dfdx(nn_approximator, x, t, order=4) + \
                            dfdx(nn_approximator, x, t, order=2) + f(nn_approximator, x, t)*dfdx(nn_approximator, x, t, order=1)


    # CHECK THE SHAPES OF OUTPUT HERE. ESPECIALLY THE LAST TERM OF THE INTERIOR_LOSS 


    # periodic boundary conditions at the domain extrema
    t_raw = torch.unique(t).reshape(-1, 1).detach().numpy()
    t_raw = torch.Tensor(t_raw)
    t_raw.requires_grad = True

    boundary_xi = torch.ones_like(t_raw, requires_grad=True) * x[0]
    boundary_loss_xi = f(nn_approximator, boundary_xi, t_raw)

    boundary_xf = torch.ones_like(t_raw, requires_grad=True) * x[-1]
    boundary_loss_xf = f(nn_approximator, boundary_xf, t_raw)


    # Neumann boundary conditions 
    neumann_loss_xi = dfdx(nn_approximator, boundary_loss_xi, t_raw, order=2)
    neumann_loss_xf = dfdx(nn_approximator, boundary_loss_xf, t_raw, order=2)

    # initial condition loss on both the function and its
    # time first-order derivative
    x_raw = torch.unique(x).reshape(-1, 1).detach().numpy()
    x_raw = torch.Tensor(x_raw)
    x_raw.requires_grad = True

    f_initial = initial_condition(x_raw)    
    t_initial = torch.zeros_like(x_raw)
    t_initial.requires_grad = True

    initial_loss_f = f(nn_approximator, x_raw, t_initial) - f_initial 
    #initial_loss_df = dfdt(nn_approximator, x_raw, t_initial, order=1)

    # obtain the final MSE loss by averaging each loss term and summing them up
    final_loss = \
        interior_loss.pow(2).mean() + \
        initial_loss_f.pow(2).mean()
    #    initial_loss_df.pow(2).mean()

    if not nn_approximator.pinning:
        final_loss += boundary_loss_xf.pow(2).mean() + boundary_loss_xi.pow(2).mean()  +\
                        neumann_loss_xi.pow(2).mean() + neumann_loss_xf.pow(2).mean()

    return final_loss
#%%

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

#%%
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

#%%
def animate_solution(nn_trained: PINN, x: torch.Tensor, t: torch.Tensor):

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    x_raw = torch.unique(x).reshape(-1, 1)
    t_raw = torch.unique(t)

    print("shape of x_raw: ", x_raw.shape)
    print("shape of t_raw:", t_raw.shape)

    def animate(i):

        if not i % 10 == 0:
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            print("shape of t partial:", t_partial.shape)
            f_train = f(nn_trained, x_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy(), f_train.detach().numpy(), label=f"Time {float(t[i])}"
            )
            ax.legend()

    n_frames = t_raw.shape[0]
    ani = FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=False)
    writergif = matplotlib.animation.PillowWriter(fps=8)
    ani.save('wave.gif',writer=writergif)
    plt.show()

#%%
def plot_solution(nn_trained: PINN, x: torch.Tensor, t: torch.Tensor):

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    ### Plot the training data set ###
    f_train = f(nn_trained, x, t)
    fig, ax = plt.subplots()
    f_train = f_train.detach().numpy()
    f_train = f_train.reshape(30, 50)
    #ax.clear()
    x_plot = torch.unique(x).reshape(-1, 1)
    t_plot = torch.unique(t)
    #ax.scatter(x_plot.detach().numpy(), f_train[:, 0], color = 'r')

    """ t_indices = np.where(t_plot>0.2)[0][0]
    print("t indices: ", t_indices)
    print(" t values: ", t_plot[t_indices])
    ax.scatter(x_plot.detach().numpy(), f_train[:, t_indices]) """
    #ax.scatter(x_plot.detach().numpy(), f_train[:, -1], color = 'g')
    ### plot a new (eval) dataset ###

    # Preparing the x and the t dimensions for evaluation
    x_eval_domain = [0.0, LENGTH]; n_points_x = 100
    t_eval_domain = [0.0, TOTAL_TIME]; n_points_t = 100
    x_eval = torch.linspace(x_eval_domain[0], x_eval_domain[1], steps=n_points_x, requires_grad=True)
    t_eval = torch.linspace(t_eval_domain[0], t_eval_domain[1], steps=n_points_t, requires_grad=True)
    grids = torch.meshgrid(x_eval, t_eval, indexing="ij")
    x_eval = grids[0].flatten().reshape(-1, 1)
    t_eval = grids[1].flatten().reshape(-1, 1)

    # making predictions on the eval data set and plotting 
    f_eval = f(nn_trained, x_eval, t_eval)
    f_eval = f_eval.detach().numpy()
    f_eval = f_eval.reshape(n_points_t, n_points_x) 
    x_plot = torch.unique(x_eval).reshape(-1, 1)
    t_plot = torch.unique(t_eval)
    #ax.plot(x_plot.detach().numpy(), f_eval[:, 0], 'r-', label='t = 0')
    t_indices = np.where(t_plot>0.2)[0][0]

    
    #ax.plot(x_plot.detach().numpy(), f_eval[:, t_indices], 'b-') 
    #ax.plot(x_plot.detach().numpy(), f_eval[:, -1], 'g-', label = 't = 1.0')
    ax.imshow(f_eval.transpose())
    #ax.set_xlabel("x (length)")
    #ax.set_ylabel("U (x, t = t)")
    ax.legend()
    plt.title("Kuramoto Sivashinsky Eqn, n = 2 (init freq)")
    plt.savefig("kuramoto_Sivashinsky.png")
    plt.show()

    return x_plot, t_plot.detach().numpy().reshape(-1,1), f_eval

#%%
from functools import partial
x_domain = [0.0, LENGTH]; n_points_x = 30
t_domain = [0.0, TOTAL_TIME]; n_points_t = 50

x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x, requires_grad=True)
t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points_t, requires_grad=True)
grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

x = grids[0].flatten().reshape(-1, 1)
t = grids[1].flatten().reshape(-1, 1)
#%%
nn_approximator = PINN(3, 15, pinning=False)
# assert check_gradient(nn_approximator, x, t)

#compute_loss(nn_approximator, x=x, t=t) # i THINK I DO NOT NEED THIS LINE

# train the PINN
loss_fn = partial(compute_loss, x=x, t=t)

nn_approximator_trained = train_model(
    nn_approximator, loss_fn=loss_fn, learning_rate=0.005, max_epochs=10000
)
  #%%
#input("Animate")
#animate_solution(nn_approximator_trained, x, t)

#%%
#input("Plot")

#saving and plotting the solution
x_eval, t_eval, u_eval = plot_solution(nn_approximator_trained,x, t)
#np.savez("kusamoto_sivashincky", x = t, t = t, u_eval = u_eval)
# %%
