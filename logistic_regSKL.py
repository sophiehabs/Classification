# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:17:03 2019

@author: Pablo
"""

import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

tot = iris.data


X = iris.data[:, :2]  # take only first two
y = (iris.target != 0) * 1

plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend();


# create model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1e20)

# train model
model.fit(X, y)

# predict on X
preds = model.predict(X)
(preds == y).mean()

print(preds)

print(model.intercept_, model.coef_)

