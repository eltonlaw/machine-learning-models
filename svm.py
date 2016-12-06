import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
mnist = fetch_mldata("MNIST original")
X_train, X_test, y_train, y_test = train_test_split(mnist.data,mnist.target,test_size=0.3,random_state=1)

# FEATURE EXTRACTION
from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# SUPPORT VECTOR CLASSIFICATION
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
predictions = clf.predict(X_test)
score = accuracy_score(y_test,predictions)
print score

