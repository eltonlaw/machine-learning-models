# AlexNet

Implementation in TensorFlow of the paper "ImageNet Classification with Deep Convolutional Neural Networks" which describes the model _AlexNet_.

### Model Summary

**NOTE:** I've tried to stay as true to the paper as possible but have still slightly modified the original implementation of AlexNet. The original implementation was trained by evenly splitting parameters onto two NVIDIA GTX 580 3GB GPUs and communication only occured in some layers. Specifically, all the kernel maps in layer 2 were used for layer 3, but the kernel maps used for layer 1->2, 3->4, 4->5 only used kernel maps from parameters on the same GPU. In this implementation, I didn't seperate the layers, so each convolutional layers volume is doubled. The likely implication of this decision is increased overfitting. Also modified is the learning rate decay, instead of a piecewise conditional learning rate decay by a factor of 10 everytime the validation error stops decreasing I used a more gradual decay: inverse time_decay with a decay rate of 0.95 every 100 0000 steps.

![](https://github.com/eltonlaw/machine_learning/blob/master/models/AlexNet/images/architecture.png?raw=true)

(from the paper (Krizhevsky et. al, 2012) [3])

* 5 Convolutional Layers, 3 Convolutional Layers
	1. Input Shape = (224x224x3)
	2. Convolutional Layer (1) 
		* Kernel Size = (11x11x3) -> 96 Channels Out
		* Stride = 4
		* Activation: ReLU
		* Local Response Normalization
			* Depth Radius = 5
			* Bias = 2
			* Alpha = 1e-4
			* Beta = 0.75
	3. Convolutional Layer (2) 
		* Kernel Size = (5x5x96) -> 256 Channels Out
		* Stride = 1
		* Activation: ReLU
		* Overlapping Max Pooling
			* Pool Size = 3x3
			* Pool Stride = 2
	4. Convolutional Layer (3)
 		* Kernel Size = (3x3x256) ->  384 Channels Out
		* Stride = 1
		* Activation: ReLU
		* Overlapping Max Pooling
			* Pool Size = 3x3
			* Pool Stride = 2
	5. Convolutional Layer (4)
		* Kernel Size = (3x3x384) ->  384 Channels Out
		* Stride = 1
		* Activation: ReLU
	6. Convolutional Layer (5)
 		* Kernel Size = (3x3x384) ->  384 Channels Out
		* Stride = 1
		* Activation: ReLU
	7. Fully Connected Layer (1)
		* 4098 Neurons
		* Activation: Relu
		* Dropout: 0.5
	8. Fully Connected Layer (2)
		* 4098 Neurons
		* Dropout: 0.5
	9. Output Layer (3)
		* 1000 Neurons (1000 classes in ImageNet)
* Optimizer: Stochastic Gradient Descent
* Weight Initialization Scheme:
	* Gaussian distribution - mean:0, stddev:0.01 for every layer
* Bias Initialization Scheme:
	* Conv2, conv4, conv5 and all fully connected layers are initialized with constant 1
	* Everything else is initialized with constant 0
* Weight decay: 0.0005
* Momentum 0.9
* Learning rate initialized at 0.01, this is divided by 10 "whenever the validation error rate stops improving with the current learning rate [3]".
* N training epochs: 90
* Batch Size: 128
* Training time on two NVIDIA GTX 580 3GB GPU's: 5-6 days

### References

[1] A. Krizhevsky. Learning multiple layers of features from tiny images. Master’s thesis, Department of Computer Science, University of Toronto, 2009.

[2] A. Krizhevsky. cuda-convnet. https://code.google.com/p/cuda-convnet/, 2012.
  
[3] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.

[4] I. Sutskever, J. Martens, G. E. Dahl, and G. E. Hinton. On the importance of initialization and momentum in deep learning. In ICML, volume 28 of JMLR Proceedings, pages 1139–1147. JMLR.org, 2013.

[5] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, pages 1929–1958, 2014.

[6] V. Nair and G. Hinton. Rectified linear units improve restricted boltzmann machines. In Proc. 27th  International Conference on Machine Learning, 2010. 

[7] Y. Boureau, J. Ponce, and Y. LeCun. A Theoretical Analysis of Feature Pooling in Visual Recognition. In International Conference on Machine Learning, 2010.

