# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:25:31 2018

@author: Neeraj
"""

import pandas as pd
import matplotlib.pyplot as plt


Salary_df = pd.read_csv("Salary.csv")

x = Salary_df.iloc[:,:-1].values
y = Salary_df.iloc[:,1].values

from sklearn.model_selection import train_test_split# split teh data into test and training
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/4, random_state=0 )

from sklearn.linear_model import LinearRegression

regressor = LinearRegression() # calling the liner regression function
regressor.fit(x_train,y_train) # passing the training values

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('training set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test),color='blue')
plt.title('set set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show


import pandas as pd
s_df = pd.read_csv("Salaries.csv", index_col = range)
x = s_df.iloc[:,:-1]

