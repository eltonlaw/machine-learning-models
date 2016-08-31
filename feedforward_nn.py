""" Feedforward Backpropogation Algorithm """
import numpy as np

# Error Settings
np.seterr(over="ignore") # To ignore the overflow error from calculating the sigmoid

### DATA PROCESSING ###

# from sklearn.datasets import fetch_mldata
# from sklearn.cross_validation import train_test_split
# mnist = fetch_mldata('MNIST original') # Labeled Handwritten digits 0-9
# x_train,x_test,y_train,y_test = train_test_split(mnist.data,mnist.target,test_size=0.0001,random_state=1)

#################
#sample data

layers = [2,3,1]
num_samples = 1
np.random.seed(seed=1)
data = np.random.randn(num_samples,layers[0])
labels = np.random.randn(num_samples,layers[-1])
x_train = data
x_test=data
y_train = labels
y_test = labels


### NEURAL NET ###

# Parameters
batch_size = len(x_train) # 1 = Stochastic GD, 1<max = mini batch, max = batch
# layers = [len(x_train[0]),400,150,9] # 9 Output layers cause we are looking for the probability of it being each value
depth = len(layers) 
biases = [np.random.randn(layer) for layer in layers[1:]] # We don't set biases for neurons in the input layer because biases are only ever used in computing the outputs from later layers
weights = [np.random.randn(output_layer,input_layer) for input_layer,output_layer in zip(layers[:-1],layers[1:])] # Returns random weights for the all the layers except the last one
alpha = 0.3 # Learning rate for gradient descent, controls the size of the step
# trained_weights = []

# Helper Functions
def sigmoid(x):
	return 1.0/(1+np.exp(-x))
def sigmoid_derivative(x):
	return sigmoid(x)*(1-sigmoid(x))
def quadratic_cost(actual,predicted): # For one training example
	return ((actual-predicted)**2)/float(2)
def total_quadratic_cost(actual,predicted): # For all training examples
	return (np.subtract(actual,predicted)**2)/float(2*len(x_train))
def quadratic_cost_derivative(actual,predicted):
	return (actual-predicted)

# used for evaluation
def forward_propogation(x):
	"""For each layer we do the dot product of the input vector and the weight vector then add the bias. The result of that is inputted into a sigmoid function. Returns a list containing the output from the final layer """
	A = x # x is the input layer for layer 1
	for W,B in zip(weights,biases): # W = weights from layer l, B = bias from layer l+1
		S = np.add(np.dot(A,np.transpose(W)),B)
		A = sigmoid(S)
	return A # This is a list which contains ten numbers representing the probabilities of each label [Pr(0),Pr(1),Pr(2)...Pr(8),Pr(9)]

def gradient_descent():
		temp_weights = []
		for i in reversed(range(len(layers)-1)):
			for j in range(layers[i]):
				print ""
				print "Layer:",i
				print "Layers[i]/Node:",j
				for weight in weights[i][j]: # For each weight in layer i,node j. Should iterate through every input nodes weights
					weight = weight - alpha*cost_derivative()
					# temp_weights.append(weight)
		# weights[i][j] = temp_weights # Simultaneously update all weights 
	

def backward_propogation(x,y): 
	# Initialize matrix of z's and matrix of activations
	zs = [np.zeros(layer) for layer in layers]
	activations = [np.zeros(layer) for layer in layers]
	
	zs[0] = x # First z's is the input
	activations[0]=sigmoid(x)

	a = sigmoid(x) # First activation is the input...Do you sigmoid the first activation?
	# Feed Forward, calculates z and activation at every node
	for w,b,i in zip(weights,biases,range(1,len(layers))): 
		z = np. add(np.dot(a,np.transpose(w)),b)
		zs[i]=z
		a = sigmoid(z)
		activations[i]=a
	# Backward propogate, calcuates error at output node and uses the weights to calculate error at every other node
	for layer in reversed(range(len(layers))): #for (n,n-1...2,1)
		if layer == len(layers)-1:
			errors = [np.zeros(layer) for layer in layers]
			errors[-1] = quadratic_cost_derivative(activations[-1],y)*sigmoid_derivative(z[-1])
		else:
			# print "SHAPES:",weights[layer-1].shape,errors[layer].shape,sigmoid_derivative(zs[layer]).shape
			errors[layer] =np.dot(errors[layer+1],weights[layer])*sigmoid_derivative(zs[layer]) # hadamard product
	print "errors:",errors


if __name__ == "__main__":
	# Uses test data
	# i = 1
	# for x,y in zip(x_test,y_test):
	# 	print "TRIAL:",i
	# 	predictions = forward_propogation(x)
	# 	a = forward_propogation
	# 	i+=1
	# 	print "Prediction:",float(np.argmax(a))
	# 	print "Actual:",y 
	# 	print ""
	backward_propogation(x_train,y_train)

