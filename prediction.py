from matplotlib import pyplot as plt
from utils import onehot_m, get_auto_mpg_data
from linear_regression import Prediction
import numpy as np


def print_pred(label_column, data, std, mean, lr):
    print(
        f"Prediction: label={data[0][label_column] * std[label_column] + mean[label_column]} prediction={lr.predict(data[0:1,1:]).item() * std[label_column] + mean[label_column]}"
    )


def train_and_test(lc, d_norm, std, mean, ax, epochs, learning_rate, lr, ylabel=""):
    print_pred(lc, d_norm, std, mean, lr)

    x_train, x_test = d_norm[100:, 1:], d_norm[:100, 1:]
    y_train, y_test = d_norm[100:, 0:1], d_norm[:100, 0:1]
    train_losses, test_losses = lr.train_test(
        x_train, x_test, y_train, y_test, epochs=epochs, learning_rate=learning_rate
    )

    print_pred(lc, d_norm, std, mean, lr)

    ax.plot(range(epochs), train_losses, label="Training loss")
    ax.plot(range(epochs), test_losses, label="Testing loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(rf"$\eta$ = {learning_rate}")
    ax.legend()


rng = np.random.default_rng()

d_raw = get_auto_mpg_data()
rng.shuffle(d_raw)
d_oh = onehot_m(d_raw, columns=[1, 7])
mean = np.mean(d_oh, axis=0)
std = np.std(d_oh, axis=0)
d_norm = (d_oh - mean) / std

lr = Prediction(d_norm.shape[1] - 1)

label_column = 0
epochs = 100
fig, axs = plt.subplots(1, 3, layout="constrained", figsize=(9, 3), sharey=True)
fig.suptitle("MSE loss vs epoch")

train_and_test(label_column, d_norm, std, mean, axs[0], epochs, 0.1, lr, ylabel="MSE")

lr.reset()
train_and_test(label_column, d_norm, std, mean, axs[1], epochs, 0.05, lr)

lr.reset()
train_and_test(label_column, d_norm, std, mean, axs[2], epochs, 0.01, lr)

fig.savefig("prediction.png")

print("Done")
