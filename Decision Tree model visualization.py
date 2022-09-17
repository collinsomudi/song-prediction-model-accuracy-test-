#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from tree import export_tree

music_data = pd.read_csv('music.csv')#importing data
X = music_data.drop(columns=['genre'])#inputs
y = music_data['genre']#outputs

model = DecisionTreeClassifier() #This is now our model we need to train it based on the data.
model.fit(X,y) #the fit method takes inputs and outputs and trains the data
plt.figure(figsize=(8, 8))
plot_tree(model,feature_names=['age','gender'], class_names=sorted(y.unique()),label='all', rounded=True, filled=True)

