import numpy as np

def f(x):
    return x**2

f = np.vectorize(f)

dx = 0.01

x_list = np.arange(-10 * dx, 10 * dx, dx)

df = np.gradient(f(x_list), 0.01)

def derivative(func, a, N, dx):
    x_list = np.arange(a - N * dx, a + N * dx, dx)
    return np.gradient(f(x_list), dx)[N]

print(derivative(f, 0.1, 10, 1))


