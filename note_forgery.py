#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:02:33 2021

@author: sid
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

## Readinf the data
df = pd.read_csv("Practical/bank.csv")

print(df.head(10))

## Checking the null values
print(df.isnull().sum())

## Checking if the data is imbalanced or not
sns.displot(df['class'],kde = False, bins= 5)


## Data is not imbalanced\
    

## Splitting into dependent and independent variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]




## Splitting the data into training and test data 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size= 0.3, random_state = 0)



## MODEL for classification
from sklearn.ensemble import RandomForestClassifier

rf  = RandomForestClassifier()
rf.fit(X_train, y_train)


## Prediction
y_pred = rf.predict(X_test)


from sklearn.metrics import accuracy_score
a_score = accuracy_score(y_test, y_pred)
print("The accuracy is:", a_score)


## Accuracy is coming arround 99%. Let's check if there's overfitting or not


## We will use cross validation
from sklearn.model_selection import cross_val_score
rf2 = RandomForestClassifier()
new_score = cross_val_score(rf2,X, y ,cv = 10 , scoring = 'accuracy').mean()
print("The updated accuracy is:", new_score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True)





## Creating a pickle file to be used later

pickle_out = open("rf.pkl", "wb")
pickle.dump(rf, pickle_out)
pickle_out.close()
