# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 19:32:14 2019

@author: Neeraj
"""

#CRIM - per capita crime rate by town
#ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#INDUS - proportion of non-retail business acres per town.
#CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#NOX - nitric oxides concentration (parts per 10 million)
#RM - average number of rooms per dwelling
#AGE - proportion of owner-occupied units built prior to 1940
#DIS - weighted distances to five Boston employment centres
#RAD - index of accessibility to radial highways
#TAX - full-value property-tax rate per $10,000
#PTRATIO - pupil-teacher ratio by town
#B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#LSTAT - % lower status of the population
#MEDV - Median value of owner-occupied homes in $1000's
#


import pandas as pd

#hdf = pd.read_csv("housing.data", header = None, delim_whitespace = True)


#dataframe_describe = hdf.describe()
#
#dataframe_matrix = hdf.corr()
#
#mode_X = hdf.mode()

#value_count = hdf["crim"].value_counts()

#Assigned column names
#hdf.isnull().sum()

X_Feature = hdf.values

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_Feature = SS.fit_transform(X_Feature)

hdf = pd.DataFrame(X_Feature)
hdf.columns = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','Istat','medv']

X = hdf.drop("medv", axis = 1) # Or
Y = hdf["medv"]




from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()


from sklearn.svm import SVR
DTR = DecisionTreeRegressor()

DTR.fit(X_train, Y_train)
Y_pred = DTR.predict(X_test)

# mse is 15.18
# accuracy is 84.82 
#mean abosulute error = 2.7803 what it is
#r2 0.7595
#mean squared log error : 0.03

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_pred, Y_test)

#MSE if we normalised all 14 columns include target value, then acuuracy of model is 99.32
#1) time : 0.15
#2 time : 0.22 accuracy score is 99%
#3) 0.39 
#4) 0.21

import seaborn as sns
import matplotlib.pyplot as plt
plt.plot(X_train, Y_test)
sns.jointplot(Y_pred,Y_test)













#hdf = le.fit_transform(X)

#X = hdf.iloc[:,:-1] #Or
#x = hdf.iloc[:,:-1].values
#for columns in hdf.columns:    
#    X = hdf.columns.values

Y = hdf["medv"]  # Or 
#Y = hdf.pop("medv") # Or

#hdf.isna().sum()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 1/4, random_state = 12345)

Or
#from sklearn.model_selection import train_test_split as tt
#x_train,x_test,y_train,y_test = tt(X, Y, test_size = 1/4, random_state = 12345)

hdf_matrix = hdf.corr()

import matplotlib.pyplot as plt

plt.title("crim with proce")
plt.xlabel("crim rate")
plt.ylabel("Price per $10000")
plt.scatter(X["crim"], Y, color = "blue")
plt.plot(X["crim"], Y, '^')
plt.show()

plt.title("residential land zone with price")
plt.xlabel("zn")
plt.ylabel("Price per $10000")
plt.scatter(X["zn"], Y, color = "red")
plt.title("crim with proce")


#dataframe_describe = hdf.describe()
#
#dataframe_matrix = hdf.corr()
#
#mode_X = hdf.mode()

#value_count = hdf["crim"].value_counts()

#Assigned column names
#hdf.isnull().sum()

X_Feature = hdf.values

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_Feature = SS.fit_transform(X_Feature)






from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)

Y_pred = regressor.predict(X_grid)

X.dtypes
Y.dtype

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

hdf = pd.read_csv("housing.data", header = None, delim_whitespace = True)

X = hdf.drop("medv", axis = 1) # Or
Y = hdf["medv"]

hdf = pd.DataFrame(X_Feature)
hdf.columns = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','Istat','medv']

X_grid = np.arange(min(X['crim']), max(X['crim']), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title()
plt.xlabel()
plt.ylabel()
plt.show()






