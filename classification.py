from matplotlib import pyplot as plt
from utils import get_wdbc_data, pred_to_class
from linear_regression import Classification
import numpy as np


def print_pred(d_x, d_y, lr):
    label = d_y[0][0]
    pred_class = pred_to_class(lr.predict(d_x[0:1, :]))
    print(f"Classification: label={int(label)} prediction={pred_class.item()}")


def train_and_test(d_x, d_y, ax, epochs, learning_rate, lr, ylabel=""):
    print_pred(d_x, d_y, lr)

    x_train, x_test = d_x[100:, :], d_x[:100, :]
    y_train, y_test = d_y[100:, :], d_y[:100, :]
    train_losses, test_losses = lr.train_test(
        x_train, x_test, y_train, y_test, epochs=epochs, learning_rate=learning_rate
    )

    print_pred(d_x, d_y, lr)

    ax.plot(range(epochs), train_losses, label="Training loss")
    ax.plot(range(epochs), test_losses, label="Testing loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(rf"$\eta$ = {learning_rate}")
    ax.legend()


rng = np.random.default_rng()

d_raw = get_wdbc_data()
rng.shuffle(d_raw)
d_x = d_raw[:, 2:]
d_y = d_raw[:, 1:2]
mean = np.mean(d_x, axis=0)
std = np.std(d_x, axis=0)
d_x_norm = (d_x - mean) / std

lr = Classification(d_x_norm.shape[1])

epochs = 1000
fig, axs = plt.subplots(1, 3, layout="constrained", figsize=(9, 3), sharey=True)
fig.suptitle("MSE loss vs epoch")

train_and_test(d_x_norm, d_y, axs[0], epochs, 0.1, lr, ylabel="MSE")

lr.reset()
train_and_test(d_x_norm, d_y, axs[1], epochs, 0.05, lr)

lr.reset()
train_and_test(d_x_norm, d_y, axs[2], epochs, 0.01, lr)

fig.savefig("classification.png")
