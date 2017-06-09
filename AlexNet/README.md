# AlexNet

Implementation in TensorFlow of the paper "ImageNet Classification with Deep Convolutional Neural Networks" which describes the model _AlexNet_.

Used on the CIFAR-10 dataset.

### Running the script

1. Clone this repository
	
		$ git clone https://github.com/eltonlaw/machine-learning-models.git
		$ cd machine-learning-models/AlexNet
		
2. Download the dataset. Run the following from command line or if you're not comfortable with that, manually download the CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and move the tar file to `../machine-learning-models/AlexNet/`

		$ curl -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
		
3. Run the model **(WIP)**
 
		$ python3 alexnet.py
		
I set the default parameters as the ones used in the paper but if you want to use different ones just pass them through, there's support for the following flags:

* `learning_rate`
* `batch_size`
* `epochs`

		$ python3 alexnet.py --learning_rate 1e-5 --batch_size=256 --epochs 100

### References

A. Krizhevsky. Learning multiple layers of features from tiny images. Master’s thesis, Department of Computer Science, University of Toronto, 2009.

A. Krizhevsky. cuda-convnet. https://code.google.com/p/cuda-convnet/, 2012.

A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.

I. Sutskever, J. Martens, G. E. Dahl, and G. E. Hinton. On the importance of initialization and momentum in deep learning. In ICML, volume 28 of JMLR Proceedings, pages 1139–1147. JMLR.org, 2013.

N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, pages 1929–1958, 2014.

V. Nair and G. Hinton. Rectified linear units improve restricted boltzmann machines. In Proc. 27th  International Conference on Machine Learning, 2010. 

Y. Boureau, J. Ponce, and Y. LeCun. A Theoretical Analysis of Feature Pooling in Visual Recognition. In International Conference on Machine Learning, 2010.
