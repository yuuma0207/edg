import math
import numpy as np
import matplotlib.pyplot as plt
import random

def U_ring5(x,y):
    divide = 0.04
    u_1 = (math.sqrt(x**2 + y**2) - 1)**2 / divide
    u_2 = (math.sqrt(x**2 + y**2) - 2)**2 / divide
    u_3 = (math.sqrt(x**2 + y**2) - 3)**2 / divide
    u_4 = (math.sqrt(x**2 + y**2) - 4)**2 / divide
    u_5 = (math.sqrt(x**2 + y**2) - 5)**2 / divide
    return min(u_1, u_2, u_3, u_4, u_5)

def sampling_from_U_ring5(n_sample):
    x_range = (-6,6)
    y_range = (-6,6)
    random_points = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n_sample)]
    energies = [math.exp(-U_ring5(x, y)) for x, y in random_points]
    plt.figure(figsize=(8, 8))
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.scatter([x for x, y in random_points], [y for x, y in random_points], c=energies, cmap='jet')
    plt.colorbar()
    plt.show()

def plot_U_ring5():
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[math.exp(-U_ring5(x, y)) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_box_aspect(1)
    c = ax.contourf(X, Y, Z, levels=100, cmap='jet')
    plt.colorbar(c, ax=ax)
    plt.show()


# sampling_from_U_ring5(10000)
plot_U_ring5()