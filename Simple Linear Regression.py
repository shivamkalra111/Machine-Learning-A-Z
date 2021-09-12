# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 23:32:58 2021

@author: shiva
"""

#simple linear regression y = b0 + b1*x1
#y is dependent variable, x is independent variable
#Ordinary least square method

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ds = pd.read_csv('Salary_Data.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,1].values

#For simple model we can have test split to be greater than normal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#Fitting simple linear regression to train set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Eperience [Training Set]')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Eperience [Test Set]')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()