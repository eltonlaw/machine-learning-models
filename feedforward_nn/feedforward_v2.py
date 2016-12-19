import numpy as np
np.random.seed(0)
import pickle

# Get Data
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")

# One hot encoding
distinct_outputs = 10
temp_labels = np.zeros((70000,distinct_outputs))
for i,label in enumerate(mnist.target):
        temp_labels[i][label] = 1
labels = temp_labels
del temp_labels

# Split data into train/test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(mnist.data,labels,test_size=0.3,random_state=1)
del labels,mnist

# Layers Array
input_layer = len(X_train[0])
output_layer = distinct_outputs
layers = [input_layer,3,output_layer] # [784,3,10]

# Create initial biases and weights
biases = [np.ones(layer)*0.1 for layer in layers[1:]]
init_weights_shape = [(node_out,node_in) for node_in,node_out in zip(layers[:-1],layers[1:])]
weights = [np.random.uniform(low=-1.0,high=1.0,size=(nodes)) for nodes in init_weights_shape]

# TRAINING
for data,label in zip(X_train,y_train):
        Z = [np.empty(layer) for layer in layers]
        # Z[0]:(784,) Z[1]:(3,), Z[2]:(10,)

# TESTING
correct = 0
total = len(y_test)
for data,label in zip(X_test,y_test):
    prediction = forward_propogation(data)
    if prediction == label:
        correct += 1.0
accuracy_score = correct/total
print(accuracy_score)

def forward_propogation(x):
    """ Feed an unknown x through the neural net and return a classification label"""
    for W_i,B_i in zip(Weights,Biases):
        z = np.matmul(W_i,x)+B_i
        a = ReLU(z)
        x = a # Sets next layer's input as this layer's activation
    output = softmax(z) # Instead of using ReLU for the last layer, use softmax
    return output


# Activation Functions
def ReLU(a):
    return max(0,x)
def ReLU_delta(a):
    if a > 0: return 1
    else: return 0
def softmax(a):
    return np.exp(a)/(np.sum(np.exp(a),axis=0))
def softmax_delta(a):

# Cost Function
def cross_entropy(a,b): 

### HELPER FUNCTIONS
# Save and read trained model parameters
def save(weights,biases,path):
    """ Save weights and biases into a pickle"""
    data = {"weights":weights,
            "biases":biases}
    with open("./output/data.pkl","wb") as o:
        pickle.dump(data,o)
def read(pickle_location):
    """_pickle_location: String path of pickle file """
    with open(pickle_location,"rb") as p:
        data = pickle.load(p)
    return data
