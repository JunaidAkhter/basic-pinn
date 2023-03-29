#%%
from typing import Callable
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from torch import nn
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#%%
# definding the parameters of the PDE
# defining the parameters
k = 2.0
m = 2.0
ω = np.sqrt(k/m)
#%%
class NNApproximator(nn.Module):
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super().__init__()

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


def f(nn: NNApproximator, x: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return nn(x)


def df(nn: NNApproximator, x: torch.Tensor = None, order: int = 1) -> torch.Tensor:
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

    print("df_value: ", df_value)

    return df_value
#%%


def compute_loss(
    nn: NNApproximator, x: torch.Tensor = None, verbose: bool = False
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss

    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    interior_loss = df(nn, x, order=2) +  ω**2 * f(nn, x)
    
    x_raw = torch.unique(x).reshape(-1, 1).detach().cpu().numpy()
    x_raw = torch.Tensor(x_raw)
    x_raw.requires_grad = True

    boundary_xi = torch.ones_like(x_raw, requires_grad=True) * x[0]

    initial_loss_f = f(nn, boundary_xi) - 1*m 
    initial_loss_df = df(nn, boundary_xi) - 2 * m 

    #final_loss = interior_loss.pow(2).mean()

    print("interior loss:", interior_loss.pow(2).mean())
    print("initial_loss_f: ", initial_loss_f.pow(2).mean())
    print("initial_loss_df:", initial_loss_df.pow(2).mean())

    final_loss = interior_loss.pow(2).mean() + initial_loss_f.pow(2).mean() + initial_loss_df.pow(2).mean()
    
    print("final_loss:", final_loss)

    return final_loss
#%%

def train_model(
    nn: NNApproximator,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
) -> NNApproximator:

    loss_evolution = []

    optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nn)
            optimizer.zero_grad()
            loss.backward()
            # Print absolute values of gradients here 
            optimizer.step()  #updates the parameters

            print("gradients:", "layer in:", torch.max(nn.layer_in.weight.grad))

            print("layer out:", torch.max(nn.layer_out.weight.grad))

            #print("layer out:", torch.max(nn.middle_layers.weight.grad))            
    
            #print("printing the gradients:", nn.weight.grad)

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

            loss_evolution.append(loss.detach().cpu().numpy())

        except KeyboardInterrupt:
            break

    return nn, np.array(loss_evolution)

#%%
def check_gradient(nn: NNApproximator, x: torch.Tensor = None) -> bool:

    eps = 1e-4
    dfdx_fd = (f(nn, x + eps) - f(nn, x - eps)) / (2 * eps)
    dfdx_sample = df(nn, x, order=1)

    return torch.allclose(dfdx_fd.T, dfdx_sample.T, atol=1e-2, rtol=1e-2)
#%%


from functools import partial

domain = [0.0, 5.0]
x = torch.linspace(domain[0], domain[1], steps=30, requires_grad=True)
x = x.reshape(x.shape[0], 1)

nn_approximator = NNApproximator(4, 50)
#assert check_gradient(nn_approximator, x)

# f_initial = f(nn_approximator, x)
# ax.plot(x.detach().cpu().numpy(), f_initial.detach().cpu().numpy(), label="Initial NN solution")

# train the PINN
loss_fn = partial(compute_loss, x=x, verbose=True)

nn_approximator_trained, loss_evolution = train_model(
    nn_approximator, loss_fn=loss_fn, learning_rate=0.1, max_epochs=20_000
)
#%%


# numeric solution
""" def logistic_eq_fn(x, y):
    return R * x * (1 - x)

numeric_solution = solve_ivp(
    logistic_eq_fn, domain, [F0], t_eval=x_eval.squeeze().detach().cpu().numpy()
) """
#%%
# plotting

x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)

fig, ax = plt.subplots()

f_final_training = f(nn_approximator_trained, x)
f_final = f(nn_approximator_trained, x_eval)

ax.scatter(x.detach().cpu().numpy(), f_final_training.detach().cpu().numpy(), label="Training points", color="red")
ax.plot(x_eval.detach().cpu().numpy(), f_final.detach().cpu().numpy(), label="NN final solution")
"""   """
ax.set(title="Periodically forced damped pendulum", xlabel="t", ylabel="f(t)")
ax.legend()

fig, ax = plt.subplots()
ax.semilogy(loss_evolution)
ax.set(title="Loss evolution", xlabel="# epochs", ylabel="Loss")
ax.legend()
plt.show()
#plt.savefig("logictic_modified.pdf")

# %%
