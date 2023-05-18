# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:30:15 2023

@author: Pablo
"""


#
#   LOAD DATA
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

print(iris.data)

#
#   SLICE INPUT DATA
#

X01 = iris.data[:, :2]  # take only 0 1
X12 = iris.data[:, 1:3]  # take only 1 2
X23 = iris.data[:, 2:4]  # take only 2 3

print(iris.target)

#
#   SLICE LABEL DATA
#

y0 = (iris.target != 0) * 1
y1 = (iris.target == 1) * 1
y2 = (iris.target != 2) * 1


#
#   PLOT DATA POINTS FUNCTION 
#

def plot_data_points(X, y):
   plt.figure(figsize=(10, 6))
   plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
   plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
   plt.legend();
   

#
#   LOGISTIC REGRESSION FUNCTION 
#
   
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()


#
#   INVOKE LOGISTIC REGRESSION FUNCTION 
#

# create model
model = LogisticRegression(lr=0.1, num_iter=300000)

# train model
model.fit(X, y)

# predict on X
preds = model.predict(X)
(preds == y).mean()


print(preds)
print(model.theta)

#
#   FUNCTION FOR PLOTTING BOUNDARIES
#

def plotting_data_with_boundary3(X,y,b,w1,w2):
    # plot result
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend()
    
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2
    
    xmin = X[:,0].min()
    xmax = X[:,0].max()
    
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    
    plt.plot(xd, yd, 'k', lw=1, ls='--')


#
#   CALLING THE FUNCTION FOR PLOTTING BOUNDARIES
#
    
plotting_data_with_boundary3(X,y,model.theta[0],model.theta[1],model.theta[2])

#
#   PRECISION AND RECALL
#

from sklearn.metrics import classification_report
print(classification_report(y, preds))

#
#   CONFUSION MATRIX
#

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y, preds)
print(cm)

#
#   TRAIN AND TEST SPLIT
#

# split into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#
#   INVOKE PLOT DATA FUNCTION
#

plot_data_points(X,y)
plot_data_points(x_train,y_train)
plot_data_points(x_test,y_test)


#
#   INVOKING SKL LOGISTIC REGRESSION FUNCTION 
#

# create model
from sklearn.linear_model import LogisticRegression

modelSKL = LogisticRegression(C=1e20)


#
#   INVOKING SKL LOGISTIC REGRESSION FUNCTION ON WHOLE SET
#

# train model
modelSKL.fit(X, y)

#
#   CHECK PREDICTION SKL LOGISTIC REGRESSION FUNCTION ON WHOLE SET
#


# predict on X
preds = modelSKL.predict(X)
(preds == y).mean()

print(preds)

print(modelSKL.intercept_, modelSKL.coef_)


#
#   INVOKING SKL LOGISTIC REGRESSION FUNCTION ON TRAIN SET
#

# train model
modelSKL.fit(x_train,y_train)

print(modelSKL.intercept_, modelSKL.coef_)

#
#   CHECK PREDICTION SKL LOGISTIC REGRESSION FUNCTION ON TEST SET
#

# predict on X
preds = modelSKL.predict(x_test)

print(preds)

#
#   PRECISION AND RECALL OVER TEST SET
#

print(classification_report(y_test, preds))

#
#   CONFUSION MATRIX  OVER TEST SET
#

cm = metrics.confusion_matrix(y_test, preds)
print(cm)

#
#   CALLING THE FUNCTION FOR PLOTTING BOUNDARIES
#


plotting_data_with_boundary3(X,y,modelSKL.intercept_[0],modelSKL.coef_.T[0],modelSKL.coef_.T[1])


#
#   SKL  SVC LINEAR
#

# train the algorithm: linear kernel
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear', degree=8)  
svclassifier.fit(x_train, y_train)  

#
#   CHECK PREDICTION SKL SVC FUNCTION ON TEST SET
#

# predict
y_predSVM = svclassifier.predict(x_test)  

#
#   PRECISION AND RECALL OVER TEST SET
#

print(classification_report(y_test, y_predSVM))


#
#   CONFUSION MATRIX  OVER TEST SET
#

cm = metrics.confusion_matrix(y_test, y_predSVM)
print(cm)

#
#   FUNCTION FOR PLOTTING BOUNDARIES AND SUPPORT VECTORS
#

def plotting_data_with_boundary_sv(X,y,b,w1,w2,support_vectors):
    # plot result
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend()
    
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2
    
    xmin = X[:,0].min()
    xmax = X[:,0].max()
    
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    
    print(support_vectors[:, 0])    
    print(support_vectors[:, 1])
    
    plt.scatter(support_vectors[:, 0],
                   support_vectors[:, 1],
                   s=30, linewidth=1, color='y', label='SV');
    
    
#
#   CALLING THE FUNCTION FOR PLOTTING BOUNDARIES ON TRAIN SET
#


plotting_data_with_boundary3(x_train,y_train,svclassifier.intercept_[0],svclassifier.coef_.T[0],svclassifier.coef_.T[1])

 
#
#   CALLING THE FUNCTION FOR PLOTTING BOUNDARIES ON TEST SET
#

plotting_data_with_boundary3(x_test,y_predSVM,svclassifier.intercept_[0],svclassifier.coef_.T[0],svclassifier.coef_.T[1])


#
#   SUPPORT VECTORS
#

print(svclassifier.support_vectors_)


#
#   CALLING THE FUNCTION FOR PLOTTING BOUNDARIES  AND SUPPORT VECTORS
#

plotting_data_with_boundary3(x_train,y_train,svclassifier.intercept_[0],svclassifier.coef_.T[0],svclassifier.coef_.T[1])

plotting_data_with_boundary_sv(x_train,y_train,svclassifier.intercept_[0],svclassifier.coef_.T[0],svclassifier.coef_.T[1], svclassifier.support_vectors_)


#
#   SKL  SVC POLY
#

svclassifier2 = SVC(kernel='poly', degree=8)  
svclassifier2.fit(x_train, y_train)  

y_predSVM2 = svclassifier2.predict(x_test)  


#
#   SKL  SVC SIGMOID
#

svclassifier3 = SVC(kernel='sigmoid')  
svclassifier3.fit(x_train, y_train)  

# predict
y_predSVM3 = svclassifier3.predict(x_test) 



#
#   SKL BAYESIAN CLASSIFIER
#

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

print(classification_report(y_test, y_pred))

cm = metrics.confusion_matrix(y_test, preds)
print(cm)