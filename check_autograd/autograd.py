import torch
import matplotlib.pyplot as plt


x = torch.linspace(0,2*3.14, 20, requires_grad=True)
x = x.reshape(-1, 1)
#f = torch.sin(x)
f = torch.tensor([[0.0000],
        [0.0226],
        [0.0451],
        [0.0676],
        [0.0901],
        [0.1125],
        [0.1349],
        [0.1572],
        [0.1795],
        [0.2016],
        [0.2237],
        [0.2456],
        [0.2674],
        [0.2890],
        [0.3106],
        [0.3319],
        [0.3531],
        [0.3741],
        [0.3950],
        [0.4156]])
print(type(f), f.shape)

df = torch.autograd.grad(f, x, torch.ones_like(x))[0]


#print("grad", df)

plt.figure()
plt.plot(x.detach().numpy(),f.detach().numpy(), "k-")
#plt.plot(x.detach().numpy(),df.detach().numpy(), "r-")
plt.show()