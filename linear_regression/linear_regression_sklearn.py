import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

mnist = fetch_mldata('MNIST original')

scores = []
skf = StratifiedKFold(n_splits=4, random_state=1)
for train_i, test_i in skf.split(mnist.data, mnist.target):
    x_tr, y_tr = mnist.data[train_i], mnist.target[train_i]
    x_te, y_te = mnist.data[test_i], mnist.target[test_i]
    lr = LinearRegression()
    lr.fit(x_tr, y_tr)
    y_pred = lr.predict(x_te)
    scores.append(accuracy_score(y_te, np.round(y_pred)))
print(np.mean(scores))
