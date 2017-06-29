import matplotlib.image as mpimg
import numpy as np
from plot_util import plot_img_before_after
from scipy.ndimage.filters import convolve


def sobel_feldman(image):
    """ A discrete differentiation operator, computes image gradients"""
    K_x = [[1, 0, -1],
           [2, 0, -2],
           [1, 0, -1]]
    K_y = np.rot90(K_x, k=3)

    G_x = convolve(image, K_x)
    G_y = convolve(image, K_y)
    # Gradient Magnitude
    G = (G_x**2 + G_y**2)**(1/2)

    plot_img_before_after(image, G)


if __name__ == "__main__":
    __FILEPATH__ = "Valve_bw.PNG"
    image = mpimg.imread(__FILEPATH__)
    sobel_feldman(image)
