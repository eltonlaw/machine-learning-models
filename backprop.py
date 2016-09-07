import numpy as np

### DATA ###
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import StratifiedKFold
mnist = fetch_mldata("MNIST original")
skf = StratifiedKFold(mnist.target,n_folds=10)

################################################################################################################
################################################################################################################
################################################################################################################

def backpropogation(x_train,y_train):
	# Loop through each datapoint
		# Feedforward and calculate error between actual and predicted. 
		# Use error at output node to calculate error at each individual node. 
		# Use errors at individual nodes to calculate gradient of weight
		# Subtract gradient*alpha from current weight
	return weights

### HELPER FUNCTIONS ###
def tanh(x):
	return np.math.tanh(x)
def tanh_derivative(x):
	return 1-np.math.pow(tanh(x),2)
def n_i(j):
	"""Normalized_initialization"""
	bound = np.sqrt(6)/float(np.sqrt(layers[j]+layers[j+1]))
	return [-bound,bound]
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
	biases = [np.zeros(layer)  for layer in layers[1:]] 
	weights = [np.random.uniform(low=n_i(j)[0],high=n_i(j)[1],size=(node_out,node_in))     for j,node_in,node_out in zip(range(len(layers)-1),layers[:-1],layers[1:])] # node_in and node_out are scalar values (784,3) and then (3,10)

	current_fold = 1
	scores = []
	for train_index,test_index in skf:
		print "FOLD:",current_fold
		x_train,x_test = mnist.data[train_index],mnist.data[test_index]
		y_train,y_test = mnist.target[train_index],mnist.data[test_index]

		trained_weights = backpropogation(x_train,y_train)
		predictions = feedforward(x_test,trained_weights)

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





