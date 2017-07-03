# Components

Random parts of machine/deep learning systems

## Filters

This is the original image.

![](https://github.com/eltonlaw/machine-learning-models/blob/master/components/images/original.png?raw=true)

### Sobel-Feldman Operator

![](https://github.com/eltonlaw/machine-learning-models/blob/master/components/images/sobel_feldman_operator.png?raw=true)

### Normalized Correlation

Kernel used is a fastened bolt:
![](https://github.com/eltonlaw/machine-learning-models/blob/master/components/images/normalized_correlation_kernel.png?raw=true)

The idea behind normalized correlation is that a good match (between the kernel and convolved partition) will create a large positive output, while everything else will create a negative output. The target output for this method was 5 white dots where the bolts are in the original image and black everywhere else. This technique doesn't seem work too well for this image/kernel pair. Possibly, the kernel is too small/not distinct enough or the original image doesn't have enough contrast. On the other hand, the output at least shows where the kernel is likely and not likely to be.

![](https://github.com/eltonlaw/machine-learning-models/blob/master/components/images/normalized_correlation.png?raw=true)
