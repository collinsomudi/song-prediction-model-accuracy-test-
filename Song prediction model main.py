#!/usr/bin/env python
# coding: utf-8

# In[83]:


#measuring accuracy of our model
#from sklearn.model_selection import train_test_split -can be used to split data into 2 sets for training and for testing
#from sklearn.metrics import accuracy_score - used fot testing the accuracy of our model.

#persisting models
import pandas as pd
from sklearn.tree import DecisionTreeClassifier#model algorithm
from joblib import dump, load#has methods for saving and loading models

music_data = pd.read_csv('music.csv')#importing data
X = music_data.drop(columns=['genre'])#inputs
y = music_data['genre']#outputs

model = DecisionTreeClassifier() #This is now our model we need to train it based on the data.
model.fit(X,y) #the fit method takes inputs and outputs
predictions = model.predict([[26, 1], [28, 0]])
predictions


# In[ ]:




