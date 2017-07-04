import numpy as np
from plot_util import Plotter


def test1():
    img = np.random.randn(100, 100)
    p = Plotter()
    p.add_to_plot(img, "image")
    p.save_plot()


def test2():
    p = Plotter()
    for i in range(15):
        img = np.random.randn(100, 100)
        p.add_to_plot(img, "image-{}".format(i))
    p.save_plot("output2.png")


if __name__ == "__main__":
    test1()
    test2()
