import numpy as np
from scipy.misc import imresize


def data_augmentation(image):
    """ Return list of 2048 sampled 224x224x3 images """
    samples = []
    for i in range(32):
        for j in range(32):
            sample_img = image[i:i+224, j:j+224, :]
            samples.append(sample_img)
            samples.append(np.rot90(sample_img, k=2))
    return samples


def resize_256x256(image):
    """ Crops and resizes all images to 256x256x3 """
    smaller_side = min(np.shape(image)[:2])
    larger_side = max(np.shape(image)[:2])
    larger_side_i = np.argmax(np.shape(image)[:2])
    i_crop = int((larger_side - smaller_side)/2)
    if larger_side_i == 0:
        image = image[i_crop:-i_crop, :, :]
    if larger_side_i == 1:
        image = image[:, i_crop:-i_crop, :]
    return imresize(image, (256, 256, 3))
