from matplotlib import pyplot as plt
from utils import *
from linear_regression import LinearRegression

rng = np.random.default_rng()

d_raw = get_data()
rng.shuffle(d_raw)
d_oh = onehot_m(d_raw, columns=[1, 7])
mean = np.mean(d_oh, axis=0)
std = np.std(d_oh, axis=0)
d_norm = (d_oh - mean) / std

lr = LinearRegression(d_norm.shape[1] - 1)

label_column = 0
print(
    f"Before training: label={d_raw[0][label_column]} prediction={lr.predict(d_norm[0:1,1:]) * std[label_column] + mean[label_column]}"
)

epochs = 100
learning_rate = 0.1
train_losses, test_losses = lr.train_test(
    d_norm, epochs=epochs, learning_rate=learning_rate
)

print(
    f"After training: label={d_raw[0][label_column]} prediction={lr.predict(d_norm[0:1,1:]) * std[label_column] + mean[label_column]}"
)

plt.plot(range(epochs), train_losses, label="Training loss")
plt.plot(range(epochs), test_losses, label="Testing loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.title("MSE loss vs epoch")
plt.savefig("lr.png")

print("Done")
