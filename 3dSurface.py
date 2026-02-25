from mpl_toolkits import mplot3d
import torch
import numpy as np
import matplotlib.pyplot as plt

def noise(shape, low=-1.0, high=1.0):
    return (high - low) * torch.rand(shape) + low

def fn(x):
    return (
        torch.sin(x[:, 0].unsqueeze(1))
        + torch.cos(x[:, 1].unsqueeze(1))
        + noise((x.shape[0], 1), low=0, high=0.6)
    ) # x[:, 0] are xs; x[:, 0] are the ys

fig = plt.figure()
ax = plt.axes(projection='3d')

grid_size = 50
x = torch.linspace(0, 5, grid_size)
y = torch.linspace(0, 5, grid_size)
X, Y = torch.meshgrid(x, y, indexing='ij')

XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

Z = fn(XY).reshape(grid_size, grid_size)

X = X.numpy()
Y = Y.numpy()
Z = Z.detach().numpy()

ax.plot_surface(X, Y, Z)

ax.set_title("3D Surface Plot")
plt.show()