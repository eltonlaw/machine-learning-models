from PIL import Image
import matplotlib.image as mpimg
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


def _get_max_indices(arr, n):
    flat = arr.flatten()
    indices = np.argsort(flat)[::-1][:n]
    indices_unraveled = np.unravel_index(indices, arr.shape, order="F")
    out = np.transpose(indices_unraveled)
    return out


def normalized_correlation(image, kernel):
    # Normalize
    image_norm = _normalize(image, 1.5, -1.5)
    correlated = correlate(image_norm, kernel)
    # max_indices = _get_max_indices(correlated, 6)
    # ks = [int((kernel.shape[0]-1)/2), int((kernel.shape[1]-1)/2)]
    # for x, y in max_indices:
    #     correlated[x-ks[0]:x+ks[0], y-ks[1]:y+ks[1]] = 0
    #     correlated[x, y] = 255
    correlated = correlated.astype(np.uint8)
    return correlated


if __name__ == "__main__":
    __FILEPATH__ = "Valve_bw.PNG"
    image = mpimg.imread(__FILEPATH__)
    imsave("./images/original.png", image)
    sobel_image = sobel_feldman(image)
    imsave("./images/sobel_feldman_operator.png", sobel_image)

    screws_kernel = image[57: 70, 68:80]
    imsave("./images/normalized_correlation_kernel.png", screws_kernel)
    nc_image = normalized_correlation(image, screws_kernel)
    imsave("./images/normalized_correlation.png", nc_image)
