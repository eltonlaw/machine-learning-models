!Feedforward Neural Network

An implementation of a standard feedforward neural network in python 3. Trained on the MNIST dataset. The weights and biases of the trained model are saved in a pickle. 

- Image data normalized to [0,1]
- Layers = [784(input),500,500,500,500,10(output, one hot encoding)]
- Using ReLU as the hidden layer activation and softmax for output layer
- Using cross entropy as the cost function
- Biases initialized at 0.1 so that ReLU neurons start off active
- Weights initiialized at sqrt(6)/(sqrt(nodes[k] + nodes[k-1]) (Glorot & Bengio, 2010)
- Optimization 

!!PLANNED

- L2 Regularization (Not implemented yet)
- Dropout (Not implemented yet)

!Sources

* https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH
* 
