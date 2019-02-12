# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 11:21:53 2018

@author: Neeraj
"""

import pandas as pd



i_df = pd.read_csv("iris.csv", header = None)

i_df.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','species']

X = i_df.iloc[:,0:4]
y = i_df.iloc[:,4]

#X = i_df.drop("species", axis = 1).values
#y = i_df["species"].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, y, random_state = 12345, test_size = 0.20)


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()

DT = DT.fit(X_train,Y_train)
Y_pred = DT.predict(X_test)

#from sklearn.metrics import accuracy_score
#ac = accuracy_score()

accuracy_score(Y_pred,Y_test)

from sklearn.model_selection import cross_validate
kf = cross_validate(DT, X, y, scoring = 'accuracy', cv = 10)



kff = kf.mean()

import matplotlib.pyplot as plt
import seaborn as sns












#Y_pred = Y_pred.astype
print(Y_pred)


import matplotlib.pyplot as plt




X_train,X_cv,Y_train,Y_cv =



from sklearn.preprocessing import train_test_split
x_train, x_test, y_train, y_test() 


describe_matrix = i_df.describe()


x = i_df.drop("species", axis  = 1).values # independent variables
y = i_df.pop("species").values # dependent varibales

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#y = ss.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=12345 )

Y_pred = 

y.value_counts()


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(70)

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf')


classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()

#y_test = le.fit_transform(y_test)
#y_pred = le.fit_transform(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
cm = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])


flower = [[4.4, 3.0, 1.0, 0.3]]
classifier.predict(flower)


