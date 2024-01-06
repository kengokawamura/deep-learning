import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


def numerical_diff(f, x):
    h = 1e-4
    return f(x + h) - f(x - h) / (2 * h)


def function_2(x):
    y = np.sum(x**2, axis=0)
    return y


# x0 = np.arange(-3, 3, 0.1)
# x1 = np.arange(-3, 3, 0.1)
# X0, X1 = np.meshgrid(x0, x1)

# y = function_2(np.array([X0, X1]))

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")

# ax.set_xlabel("x0")
# ax.set_ylabel("x1")
# ax.plot_wireframe(X0, X1, y)
# plt.show()


class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()


f = lambda w: net.loss(x, t)

dw = numerical_gradient(f, net.W)

print(dw)
