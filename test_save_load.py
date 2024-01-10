from linear_regression import LinearRegression
import numpy as np


def test_save_load():
    file = "test.npy"

    lr = LinearRegression(3)
    w = lr.w
    lr.save(file)

    lr_new = LinearRegression()
    lr_new.load(file)
    w_new = lr_new.w

    assert np.array_equal(w, w_new)
