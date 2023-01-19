import torch

import matplotlib.pyplot as plt

steps = 4
xs = torch.linspace(-5, 5, steps=steps)
ys = torch.linspace(-5, 5, steps=steps)
x, y = torch.meshgrid(xs, ys, indexing='xy')
z = (x * x + y * y)

ax = plt.axes(projection='3d')

print("x:", x.numpy())
print("y:", y.numpy())
print("z:", z.numpy())
ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
plt.show()