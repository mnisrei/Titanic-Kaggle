#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:47:40 2019

@author: rei
"""
import pandas as pd
import numpy as np
# Importing the dataset
dataset = pd.read_csv('train.csv')
#dataset['Embarked']
dataset = dataset.drop(['Name','Ticket','Cabin','Embarked','PassengerId'], axis = 1)
dataset = dataset[['Sex', 'Pclass','Age', 'Parch','SibSp','Fare','Survived']].fillna(0)
mean_age = int(dataset['Age'].dropna().mean())
dataset.loc[dataset.Age == 0, 'Age'] = mean_age
X = dataset.iloc[:, :6].values
y = dataset.iloc[:, 6].values
############################
#Label
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
print(X)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.50)
