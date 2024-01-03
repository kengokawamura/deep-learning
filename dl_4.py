import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size


def numerical_diff(f, x):
    h = 1e-4
    return f(x + h) - f(x - h) / (2 * h)


def function_2(x):
    y = np.sum(x**2, axis=0)
    return y


x0 = np.arange(-3, 3, 0.1)
x1 = np.arange(-3, 3, 0.1)
X0, X1 = np.meshgrid(x0, x1)

y = function_2(np.array([X0, X1]))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.plot_wireframe(X0, X1, y)
plt.show()
