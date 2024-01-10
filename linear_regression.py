import numpy as np


def validate(f):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "w") or not hasattr(self, "b"):
            raise Exception("Model is not initialized")

        return f(self, *args, **kwargs)

    return wrapper


class LinearRegression:
    def __init__(self, m=None):
        if m:
            self.__reset(m)

    def __loss_n(self, x, y):
        pred = np.matmul(x, self.w) + self.b
        l = np.mean((pred - y) ** 2, axis=0)

        return l.item()

    def __dloss_n(self, x, y):
        pred = np.matmul(x, self.w) + self.b
        dl_w = np.mean(2 * (pred - y) * x, axis=0, keepdims=True)
        dl_b = np.mean(2 * (pred - y), axis=0)

        return (dl_w.T, dl_b.item())

    def __reset(self, m):
        rng = np.random.default_rng()

        self.w = rng.random((m, 1))
        self.b = rng.random()

    @validate
    def reset(self):
        self.__reset(self.w.shape[0])

    @validate
    def train_test(self, data, epochs=100, learning_rate=0.01):
        if data.shape[1] - 1 != self.w.shape[0]:
            raise Exception(f"Invalid number of features {data.shape[1]}")

        train_losses = []
        test_losses = []
        x_train, x_test = data[100:, 1:], data[:100, 1:]
        y_train, y_test = data[100:, 0:1], data[:100, 0:1]

        for epoch in range(epochs):
            loss_train_epoch = self.__loss_n(x_train, y_train)
            train_losses.append(loss_train_epoch)

            dl_w, dl_b = self.__dloss_n(x_train, y_train)
            self.w = self.w - learning_rate * dl_w
            self.b = self.b - learning_rate * dl_b

            loss_test_epoch = self.__loss_n(x_test, y_test)
            test_losses.append(loss_test_epoch)

            print(
                f"Epoch: {epoch + 1} Train loss: {train_losses[epoch]} Test loss: {test_losses[epoch]}"
            )

        return (train_losses, test_losses)

    @validate
    def predict(self, x):
        y = np.matmul(x, self.w) + self.b

        return y.item()

    @validate
    def save(self, file):
        w = np.append(self.w.squeeze(), self.b)
        with open(file, "wb") as f:
            np.save(f, w)

    def load(self, file):
        with open(file, "rb") as f:
            w = np.load(f)
            self.w = w[:-1, np.newaxis]
            self.b = w[-1]
