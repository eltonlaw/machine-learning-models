import numpy as np

### SETTINGS ###
np.random.seed(1)

### DATA ###
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import StratifiedKFold
mnist = fetch_mldata("MNIST original")
skf = StratifiedKFold(mnist.target,n_folds=10)

################################################################################################################
################################################################################################################
################################################################################################################

def backward_prop(x_train,y_train,initial_weights,initial_biases):
	"""lnitialize matrix of z's and atrix of activations"""
	weights = initial_weights
	biases = initial_biases
	zs = [np.zeros(layer) for layer in layers]
	activations = [np.zeros(layer) for layer in layers]
	counter = 1
	for x,y in zip(x_train,y_train):
		x = np.array(x)
		y = np.array(y)
		print "TRAINING TRIAL:",counter
		
		# First layer, set input layer as the data and create activations from that
		zs[0] = x # First z's is the input vector
		activations[0]=tanh(x) 
		a = activations[0]	
		# Feed Forward, calculates z and activation at every node
		for w,b,i in zip(weights,biases,range(1,len(layers))): 
			if  i <len(layers)-1: # For all hidden layers compute the z and activation
				z = np.add(np.dot(a,np.transpose(w)),b)
				zs[i]=z
				a = tanh(z)
				activations[i]=a
			elif i == len(layers)-1: # For the last layer use 
				z=np.add(np.dot(a,np.transpose(w)),b)
				zs[i]=z
				a=softmax(z) # For our final layer we want to output a probability
				activations[i]=a
				predictions = activations[i]

		# Backward propogate, calcuates error at output node and uses the weights to calculate error at every other node
		for layer in reversed(range(len(layers))): #for (n,n-1...2,1)
			if layer == len(layers)-1: # If this is the first layer, create the errors and set the first error layer
				errors = [np.zeros(layer) for layer in layers]
				errors[-1] = cross_entropy(predictions,y,len(x_train))
			else:
				print errors[layer+1]
				errors[layer] =np.dot(errors[layer+1],weights[layer])*tanh_derivative(zs[layer]) # hadamard product

		# Find the change in Cost attributable to a change in weight
		gradients = weights # To get the same dimensionality as the weights list
		for l in range(len(layers)-1):
			for k in range(layers[l]): # K is input layer
				for j in range(layers[l+1]): # J is output layer
					print "activations[l][k]:",activations[l][k]
					print "errors[l+1][j]:",errors[l+1]
					gradients[l][j][k] = activations[l][k]*errors[l+1][j] #First term is partial derivative of cost with respect to weight, Second term is the regularization term
		# Change weights according to gradients [weights = weights - (learning rate * delta)]
		weights = np.subtract(weights,np.multiply(gradients,learning_rate))
		counter += 1
	return weights

def forward_propogation(x_train,weights,biases):
	"""For each layer we do the dot product of the input vector and the weight vector then add the bias. The result of that is inputted into a sigmoid function. Returns a list containing the output from the final layer """
	predictions = []# to get the same shape as x_train
	for x in x_train:
		A = x # x is the input layer for layer 1
		for W,B in zip(weights,biases): # W = weights from layer l, B = bias from layer l+1
			S = np.add(np.dot(A,np.transpose(W)),B)
			A =tanh(S)
		predictions.extend(A)
	return predictions # This is a list which contains ten numbers representing the probabilities of each label [Pr(0),Pr(1),Pr(2)...Pr(8),Pr(9)]
### HELPER FUNCTIONS ###
def tanh(z):
	return np.tanh(z)
def tanh_derivative(z):
	z = tanh(z)
	if isinstance(z,int): # If it's an integer 
		return 1-np.math.pow(z,2)
	else: # If it's a list, enumerate through the list and apply square to each one
		for z_i,i in enumerate(z):
			z[i] = 1-np.math.pow(z_i,2)
		return z
def softmax(z):
	e=np.exp(z-np.max(z))
	return e/np.sum(e)
def n_i(j):
	"""Normalized_initialization for randomized weights"""
	bound = np.sqrt(6)/float(np.sqrt(layers[j]+layers[j+1]))
	return [-bound,bound]
def cross_entropy(y_pred,y,length_of_training):
	# Returns average cross entropy error of 1 set of data
	cost=-np.log(y_pred[int(y)])/float(length_of_training)
	return cost
def accuracy_score(predictions,actuals): 
	""" Compares the predicted most probable value to the actual value and if they're equal, total_correct+=1. 
	__predictions__: A list of lists. Contains the probabilities of the specific input actually being that value. Ex. predictions = [[0.145,0.2545,0.123,0.3432,0.12312,0.1231,0.123...],[...]...]
	__actuals__: A list of values. Contains what value the input actually is. Ex. actuals = [0,1,3,1,4,6,9,5,7,6...]
	"""
	total = 0
	total_correct = 0
	for jj,prediction in enumerate(predictions): 
		if prediction.index(max(prediction)) == actuals[jj]:
			total_correct +=1
		total +=1
	print "CORRECT:",total_correct
	print "TOTAL:",total
	print "PERCENTAGE CORRECT:",total_correct/float(total)

################################################################################################################
################################################################################################################
################################################################################################################

if __name__ == "__main__":
	#Parameters
	layers = [mnist.data.shape[1],3,len(set(mnist.target))] # Input layer is the matrix of pixels[28*28]. Output is the amount of unique y labels[0-9].
	initial_biases = [np.zeros(layer)  for layer in layers[1:]] 
	initial_weights = [np.random.uniform(low=n_i(j)[0],high=n_i(j)[1],size=(node_out,node_in))     for j,node_in,node_out in zip(range(len(layers)-1),layers[:-1],layers[1:])] # node_in and node_out are scalar values (784,3) and then (3,10)
	learning_rate = 0.3 # Initial learning rate 

	current_fold = 1
	scores = []

	for train_index,test_index in skf:
		print "FOLD:",current_fold
		x_train,x_test = mnist.data[train_index],mnist.data[test_index]
		y_train,y_test = mnist.target[train_index],mnist.data[test_index]

		trained_weights = backward_prop(x_train,y_train,initial_weights,initial_biases)
		predictions = forward_prop(x_test,trained_weights)

		scores.append(accuracy_score(predictions,y_test))
		current_fold+=1
	print "AVERAGE FINAL ACCURACY:",np.mean(scores)
	

""" New changes to implement

- StratifiedKFold
- Weights initialized 
- Biases initialized at 0
- Find a different cost function
- Output is the probability of being each number
- Activation function: Hyperbolic tangent(tanh)
- 

"""





