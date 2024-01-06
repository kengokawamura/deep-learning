import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
from two_layer_net import TwoLayerNet
from sample_code.mnist import load_mnist


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


# x = np.array([0.6, 0.9])
# t = np.array([0, 0, 1])

# net = simpleNet()


# f = lambda w: net.loss(x, t)

# dw = numerical_gradient(f, net.W)

# print(dw)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# ハイパーパラメータ

iters_num = 10000
batch_size = 100
learning_rate = 0.1
train_size = x_train.shape[0]

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配
    grad = network.numerical_gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] = learning_rate * grad[key]

    # 学習結果の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(f"今 #{i+1} 回")

# グラフの描画
print(train_loss_list)
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label="loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()
