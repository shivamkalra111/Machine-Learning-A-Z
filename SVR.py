# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 20:13:39 2021

@author: shiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

#Reading the data
ds = pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:,1:2].values
y = ds.iloc[:, 2:].values

# Feature Scaling [SVR does not have feature scaling in its algorithm]
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to dataset
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
#For this we had to transform as data was scaled for regressor, we are applying inverse transform for getting value
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))


# Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff [SVR]')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

#For getting curve we can use grid method from Polynomial Regression

