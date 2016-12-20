import numpy as np
np.random.seed(0)
import pickle
import time
import sys

### HELPER FUNCTIONS
# Save and read trained model parameters
def save(weights,biases,path="./output/data.pkl"):
    """ Save weights and biases into a pickle"""
    data = {"weights":weights,
            "biases":biases}
    with open(path,"wb") as o:
        pickle.dump(data,o)
    print("Saved pickle to {}".format(path))
def read(pickle_location):
    """_pickle_location: String path of pickle file """
    with open(pickle_location,"rb") as p:
        data = pickle.load(p)
    return data

## ACTIVATION FUNCTIONS
def ReLU(A):
    for ii,a_i in enumerate(A):
        A[ii] = max(0.0,a_i)
    return A
def ReLU_delta(A):
    for ii,a_i in enumerate(A):
        if a_i > 0: A[ii] = 1
        else: A[ii] = 0
    return A
def softmax(A):
    return np.exp(A)/(np.sum(np.exp(A)))

# Cost Function
def cross_entropy(A,B): 
    total = 0
    for a_i,b_i in zip(A,B):
        total -= b_i*np.log(a_i)
    return total

def forward_propogation(x):
    """ Feed an unknown x through the neural net and return a classification label"""
    for W_i,B_i in zip(weights,biases):
        z = np.matmul(W_i,x)+B_i
        a = ReLU(z)
        x = a # Sets next layer's input as this layer's activation
    output = softmax(z) # Instead of using ReLU for the last layer, use softmax
    return output

def alpha(t):
    return 1/(0.5*t)**2

##################################
# Get Data
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")

# One hot encoding
distinct_outputs = 10
temp_labels = np.zeros((70000,distinct_outputs))
for i,label in enumerate(mnist.target):
    temp_labels[i][label] = 1.0
labels = temp_labels
del temp_labels

# Split data into train/test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(mnist.data,labels,test_size=0.3,random_state=1)
del labels,mnist

# Layers Array
input_layer = len(X_train[0])
output_layer = distinct_outputs
layers = [input_layer,5,output_layer] # [784,3,10]

# Create initial biases and weights
biases_shape = layers[1:]
biases = [np.ones(layer)*0.1 for layer in biases_shape]
weights_shape = [(node_out,node_in) for node_in,node_out in zip(layers[:-1],layers[1:])]
weights = [np.random.uniform(low=-np.sqrt(6)/(np.sqrt(nodes[1]+nodes[0])),high=np.sqrt(6)/(np.sqrt(nodes[1]+nodes[0])),size=(nodes)) for nodes in weights_shape] 

# TRAINING
t = 1.0
start_time = time.perf_counter() 
for data,label in zip(X_train,y_train):
    # Calculate hidden layer z's and activations
    zs = [np.empty(layer) for layer in layers[1:]] #  [0]:(3,), [1]:(10,)
    activations = [np.empty(layer) for layer in layers] # [0]:(784,) [1]:(3,), [2]:(10,)
    activations[0] = data/255.0
    for i,n_nodes in enumerate(layers[1:]): 
        zs[i] = np.matmul(weights[i],activations[i]) + biases[i]
        activations[i+1] = ReLU(zs[i])
    activations.pop(0)
    activations[-1] = softmax(zs[-1]) # Recalculate and overwrite the last ReLU layer/output layer with a softmax layer
    # Compute partial_derivative
    partial_derivative = [np.empty(shape) for shape in (weights_shape+[output_layer]) ] # [(5,784),(10,5),(10,)] - (output,input) 
    partial_derivative[2] = -(label - activations[1]) # (10,)
    temp = np.transpose(weights[1])*partial_derivative[2]
    partial_derivative[1] = [np.sum(node) for node in temp]
    temp = np.transpose(weights[0])*partial_derivative[1]
    partial_derivative[0] = [np.sum(node) for node in temp]
    # Calculate weight gradients
    weight_gradients = [np.empty(shape) for shape in weights_shape]
    for i,(pd,a) in  enumerate(zip(partial_derivative[:-1],activations)):
        pd,a = [pd],[a]
        weight_gradients[i] = np.matmul(np.transpose(a),pd)
    # Calculate bias gradients
    bias_gradients = [np.empty(shape) for shape in biases_shape]
    for i,pd in  enumerate(partial_derivative[1:]):
        bias_gradients[i] = pd * 1
    # Update
    weights += np.multiply(weight_gradients,alpha(t))
    biases += np.multiply(bias_gradients,[np.full_like(b,alpha(t)) for b in bias_gradients])
    t += 1
    progress = (t*100.00)/len(X_train) 
    if progress % 0.5 == 0:
        sys.stdout.write("Percentage Trained [ {:.2f}% ] \r".format(progress))
        sys.stdout.flush()
time_elapsed = time.perf_counter() - start_time
print("Training finished. Time Elapsed: {}".format(time_elapsed))
save(weights,biases)
# TESTING
correct = 0
total = len(y_test)
for data,label in zip(X_test,y_test):
    prediction = forward_propogation(data)
    if label[np.argmax(prediction)] == 1:
        correct += 1.0
accuracy_score = correct/total
print(accuracy_score)

