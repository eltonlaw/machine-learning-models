import numpy as np


def _convolve(image, focus_i, kernel):
    """
    image
    focus_i: Indices
    kernel
    """
    kr = int((np.shape(kernel)[0]-1)/2)
    x, y = focus_i
    return np.sum(np.multiply(image[x-kr:x+kr+1, y-kr:y+kr+1], kernel))


def cnn_convolution(image, kernel, stride, padding):
    """ Discrete Convolution only using square kernels, centre focused,
    zero padding

    padding: ["SAME", "FULL", "VALID"]
        - "SAME" means that the dimension of the output is the same as
        the dimension of the input
        - "FULL" means every pixel is convolved the same number f times
        - "VALID" means that you only apply convolution in a way so that
        no extra padding is required
    """
    k_n = np.shape(kernel)
    kr = int((k_n[0] - 1)/2)
    original_image_shape = np.array(np.shape(image))

    assert k_n[0] == k_n[1]  # Kernel has to be square
    assert k_n[0] % 2 == 1  # Kernel has to be odd numbered

    # Padding and output shape varies depending on method
    if padding == 'FULL':
        padding = k_n[0] - 1
        new_shape = original_image_shape/stride + kr
    elif padding == 'SAME':
        padding = kr
        new_shape = original_image_shape/stride
    elif padding == 'VALID':
        padding = 0
        new_shape = original_image_shape/stride - kr
    new_shape = np.ceil(new_shape).astype(np.int)

    image = np.pad(image, padding, 'constant', constant_values=0)
    image_shape = np.shape(image)

    convolved = []
    for x in range(int(np.floor(image_shape[0]/stride))):
        for y in range(int(np.floor(image_shape[1]/stride))):
            val = _convolve(image, (x*stride+kr, y*stride+kr), kernel)
            convolved.append(val)

#    for x in range(int(np.ceil(original_image_shape[0]/stride))):
#        for y in range(int(np.ceil(original_image_shape[1]/stride))):
#            val = _convolve(image, (x*stride+kr, y*stride+kr), kernel)
#            convolved.append(val)

    convolved = np.reshape(convolved, new_shape)
    return convolved


def test_cnn_convolution():
    image = np.ones((5, 5))
    kernel = np.random.randn(3, 3)
    stride = 2
    for padding in ["SAME", "VALID", "FULL"]:
        print(cnn_convolution(image, kernel, stride, padding))


def general_convolution(image, kernel, kernel_position, stride, padding):
    """ Convolution with non-square kernels

    kernel_position: Must be a valid index in the parameter `kernel`
       The position in which each kernel is "focused"
    """
    pass


if __name__ == "__main__":
    test_cnn_convolution()
