# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:00:16 2019

@author: Pablo
"""


import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
X, y = make_circles(100, factor=.1, noise=.1)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');