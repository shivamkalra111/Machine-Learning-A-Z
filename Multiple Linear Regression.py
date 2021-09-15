# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 21:51:45 2021

@author: shiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#Dummy variable - Trap, duplicating variable as d2 = 1 - d1
'''
Step-wise regression model - 
1. All in [Use all variables only if prior knowledge or forced]
2. Backward Elimination [Selecting significance level (SL == 0.05, if P>SL remove it)]
3. Forward Selection [SL = 0.05, use simple regression with all, keeping lowest P(Probability) value and keep adding linear regressions until P>SL]
4. Bidirectional Elimination
5. All Possible Models (2^n)-1 combinations [not a good approach]
'''

# We are using backward elimination because it is faster

ds = pd.read_csv('50_Startups.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:, 4].values

# Encoding Categorical Data
##Encoding the Independent Variable
onehotencoder = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = "passthrough")
X = np.array(onehotencoder.fit_transform(X), dtype = np.float)

#Avaoiding Dummy variable Trap
X = X[:, 1:] # No need to do that, Python libraries take care of the dummy variable trap


#splitting train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# No feature Scaling for Multiple Linear regression as Libraries will take care of the same

#Fitting Multiple Linear Regression to the Training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediction of the test set result
y_pred = regressor.predict(X_test)

#Building Optimal Model using Backward Elimination
#Adding a column of 1s as variable x0 which has b0 constant
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()# Endog is dependent variable
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]] # Removed 2 as it has the max P value that was over 5%
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()# Endog is dependent variable
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]] # Removed 1 as it has the max P value that was over 5%
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()# Endog is dependent variable
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]] # Removed 4 as it has the max P value that was over 5%
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()# Endog is dependent variable
regressor_OLS.summary()

X_opt = X[:, [0, 3]] # Removed 5 as it has the max P value that was over 5%
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()# Endog is dependent variable
regressor_OLS.summary()


X_opt = X_opt[:,1:]

#splitting train and test set
X_train_opt, X_test_opt, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training Set
regressor = LinearRegression()
regressor.fit(X_train_opt, y_train)

#Prediction of the test set result
y_pred_opt = regressor.predict(X_test_opt)

#Visualising the training set results
plt.scatter(X_train_opt, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train_opt), color = 'blue')
plt.title('Marketing Spend vs Profit [Training Set]')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()

plt.scatter(X_test_opt, y_test, color = 'red')
plt.plot(X_train_opt, regressor.predict(X_train_opt), color = 'blue')
plt.title('Marketing Spend vs Profit')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()