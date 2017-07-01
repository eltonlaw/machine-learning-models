from PIL import Image
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import correlate
from scipy.misc import imsave


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
    return G


def _normalize(img, new_max, new_min):
    i_min = np.min(img)
    i_max = np.max(img)
    new_img = (img - i_min)*(new_max - new_min)/(i_max - i_min) + new_min
    return new_img


def normalized_correlation1(image, kernel):
    # Normalize
    image_norm = _normalize(image, 255, -255)
    np.min(image_norm)
    np.max(image_norm)
    correlated = correlate(image_norm, kernel)
    np.min(correlated)
    return correlated


if __name__ == "__main__":
    __FILEPATH__ = "Valve_bw.PNG"
    image = np.array(Image.open(__FILEPATH__))

    sobel_image = sobel_feldman(image)
    imsave("sobel_feldman_operator.png", sobel_image)

    screws_kernel = image[57: 70, 68:80]
    imsave("normalized_correlation_kernel.png", screws_kernel)
    nc_image = normalized_correlation1(image, screws_kernel)
    imsave("normalized_correlation.png", nc_image)
