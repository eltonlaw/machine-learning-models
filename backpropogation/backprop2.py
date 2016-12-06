import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
mnist = fetch_mldata("MNIST original")

labels = np.zeros((len(mnist.target),len(np.unique(mnist.target))))
for ii,label in enumerate(mnist.target):
	labels[ii][label] = 1
print labels

x_train,x_test,y_train,y_test = train_test_split(mnist.data,labels,test_size=0.3,random_state=1)


############################

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2, activity_l2

model= Sequential()
model.add(Dense(32,input_dim=784))
model.add(Dense(1000,activation="tanh",W_regularizer=l2(0.01)))
model.add(Dense(1000,activation="tanh",W_regularizer=l2(0.01)))
model.add(Dense(1000,activation="tanh",W_regularizer=l2(0.01)))
model.add(Dense(1000,activation="tanh",W_regularizer=l2(0.01)))
model.add(Dense(1000,activation="tanh",W_regularizer=l2(0.01)))
model.add(Dense(10,activation="softmax",W_regularizer=l2(0.01),activity_regularizer=activity_l2(0.01)))


model.compile(optimizer="sgd",loss="categorical_crossentropy",metrics=['accuracy'])

print "x_train:",x_train.shape
print "y_train:",y_train.shape


# Training
model.fit(x_train,y_train,nb_epoch=1,batch_size=32)



# Testing
score = model.evaluate(x_test,y_test)
print score

