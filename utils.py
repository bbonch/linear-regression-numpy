import numpy as np
import pandas as pd


def onehot_x(x, max, min):
    z = np.zeros(int(max - min + 1))
    z[int(x - min)] = 1

    return z


def onehot_v(v):
    unique_v = np.unique(v)
    r = np.array([onehot_x(x, unique_v[-1], unique_v[0]) for x in v])
    r = r[:, np.any(r != 0, axis=0)]

    return r


def onehot_m(m, columns: list):
    r = []
    for i in range(m.shape[1]):
        if i in columns:
            r.append(onehot_v(m[:, i]))
        else:
            r.append(m[:, i : i + 1])
    r = np.concatenate((r), axis=1)

    return r


def get_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    dataset = pd.read_csv(
        url, na_values="?", comment="\t", header=None, sep=" ", skipinitialspace=True
    )
    dataset = dataset.dropna()

    dataset_np = dataset.to_numpy()

    return dataset_np
