#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier#model algorithm
from sklearn.model_selection import train_test_split#algorithim for testing and training 
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')#importing data
X = music_data.drop(columns=['genre'])#inputs
y = music_data['genre']#outputs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size=0.8)
model = DecisionTreeClassifier()#This is now our model we need to train it based on the data. model.fit(X_train,y_train) #the fit method takes inputs and outputs
model.fit(X_train, y_train)
predictions = model.predict(X_test)#ask model to make two predictions at once ie boy 21 and girl 22 which songs will they like?
score = accuracy_score(y_test, predictions)
score


