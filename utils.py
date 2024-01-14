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


def sigmoid(x):
    r = 1.0 / (1.0 + np.exp(-x))
    r[r == 1.0] = 0.9999
    r[r == 0.0] = 0.0001

    return r


def pred_to_class(pred, treshold=0.5):
    return (pred > treshold).astype(int)


def get_auto_mpg_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    dataset = pd.read_csv(
        url, na_values="?", comment="\t", header=None, sep=" ", skipinitialspace=True
    )
    dataset = dataset.dropna()

    dataset_np = dataset.to_numpy()

    return dataset_np


def get_wdbc_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    dataset = pd.read_csv(url)
    dataset.iloc[:, 1] = dataset.iloc[:, 1].map({"B": 0, "M": 1})

    dataset_np = dataset.to_numpy(dtype=np.float32)

    return dataset_np
