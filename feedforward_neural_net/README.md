# About

All models trained on the MNIST dataset.

### 'feedforward_tf.py

- Weights initialized using a truncated normal distribution - N(0, 1)
- Biases initialized at 0.1
- Cost function: Cross Entropy
- Layers:
	- Input - 784,
	- Dense - 500 - relu
	- Dense - 300 - relu,
	- Output - 10 - softmax - one hot encoding
- Learning Rate: 
	- Start at 0.02
	- 0.95 decay rate
	- 20000 decay steps
- Optimizer: RMSProp
- Batch Size = 256
- Training Epochs = 100

### 'feedforward.py' and 'feedforward_v2.py'
An implementation of a standard feedforward neural network in vanila python 3. The weights and biases of the trained model are saved in a pickle. 

- Image data normalized to [0,1]
- Layers = [784(input),500,500,500,500,10(output, one hot encoding)]
- Using ReLU as the hidden layer activation and softmax for output layer
- Using cross entropy as the cost function
- Biases initialized at 0.1 so that ReLU neurons start off active
- Weights initiialized at sqrt(6)/(sqrt(nodes[k] + nodes[k-1]) (Glorot & Bengio, 2010)
- Trained with backprop 

# References
* https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH
* http://neuralnetworksanddeeplearning.com/
* http://www.deeplearningbook.org/
