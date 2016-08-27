import numpy as np
import random

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

### DATA PROCESSING ###

mnist = fetch_mldata('MNIST original')


x_train,x_test,y_train,y_test = train_test_split(mnist.data,mnist.target,test_size=0.3,random_state=1)

print x_train.shape,x_test.shape,y_train.shape,y_test.shape

### NEURAL NET ###

# Parameters
"""
layers = [8,5,9,4]
sample_input = np.random.randn(layers[0])

depth = len(layers) # Depth = # of layers
layers = layers
biases = [np.random.randn(layer) for layer in layers[1:]] # We don't set biases for neurons in the input layer because biases are only ever used in computing the outputs from later layers


weights = [np.random.randn(input_layer,output_layer) for input_layer,output_layer in zip(layers[:-1],layers[1:])] # Returns random weights for the all the layers except the last one

# Function to produce an S-shaped curve
def sigmoid(x):
	return 1.0/(1+np.exp(-x))
def sigmoid_derivative(x):
	return sigmoid(x)*(1-sigmoid(x))

# 
def forward_propogation(x):
	A = x
	i = 0
	for W,B in zip(weights,biases):

		print "LAYER:",i
		print "A:",A
		print "W",W
		print "B",B
		print "Z:",np.dot(A,W)
		s = np.add(np.dot(A,W),np.transpose(B)) # B is in shape (3,1), and the dot product of A,W is in shape (1,3)
		print "s:",s
		A = sigmoid(s)
		print "A:",A
		i+=1
	return A

if __name__ == "__main__":
	# Debug
	print "depth:",depth
	print "layers:",layers
	print "biases:",biases
	print "biases_shape:",np.shape(biases)
	print "weights:",weights
	print "weights_shape:",np.shape(weights[0])

	print "weights[0][0][1]:",weights[0][0][1]
	print ""
	forward_propogation(sample_input) 
"""
