# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 22:28:25 2021

@author: shiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Reading the data
ds = pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:,1:2].values
y = ds.iloc[:, 2].values

#No need for feature scaling and train_tes_split

# Fitting Linear Regression to dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree = 4)#Creates a column of a constant, increae degree for better results
X_Poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_Poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff [Linear Regression]')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()


# Visualizing the Polynomial Regression Results
plt.scatter(X, y)
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')# For new data
plt.title('Truth or Bluff [Polynomial Regression]')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

#degree 4 is best for now, as it is getting best options
# Although this is giving straight lines b/w the points, so we do in comments below
'''
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)#no. of lines is no. of shape of X_grid, no. of columns
plt.scatter(X, y)
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')# For new data
plt.title('Truth or Bluff [Polynomial Regression]')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()
'''

# Predicting new results with Linear Regression
lin_reg.predict([[6.5]])#not a good model


# Predicting new results with Polynomial Regression\
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))