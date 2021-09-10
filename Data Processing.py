# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 22:44:39 2021

@author: shiva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# X has independent variables while y has dependent
ds = pd.read_csv('Data.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,3].values

#For imputing(filling) missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


#LabelEncoder is not used for OneHotEncoder
#OneHotEncoder is used for dummy variables so that each value has same weightage
onehotencoder = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = "passthrough")
X = np.array(onehotencoder.fit_transform(X), dtype = np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#X = onehotencoder.fit_transform(X) ''' This converts to array of objects but we need array of float '''
#Label encoder is used for independent variable

#Splitting dataset into 2 datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling is required as euclidean distance can be dominated by any variable
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
# No feature scaling on categorical dependent variable so not on y




