#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:01:29 2019

@author: rei
"""
from keras.models import model_from_yaml
import pandas as pd
#model loaded and used
def model(X):
####load Yaml######
    yaml_file = open("titanic_model.yaml", "r")
    loaded_model = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model)
    predict_test = loaded_model.predict(X)
    if predict_test>0.5:
        return True
    else:
        return False
##############################################################################
#data preprocessing

def datasets(dataset):
    dataset = dataset[['Sex', 'Pclass','Age', 'Parch','SibSp','Fare']].fillna(0)
    X = dataset.iloc[:, :6].values
    print(X)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = sc.transform(X)
    return X
#############################
#user input
def input_user():
    sex = input("gender: ")
    Pclass = input("Pclass: ")
    Age = input("Age: ")
    Parch = input("Parch: ")
    SibSp = input("SibSp: ")
    Fare = input("Fare: ")
    if sex.lower() == 'male':
        sex = 1
    else:
        sex = 0
    data = [[sex,Pclass,Age,Parch,SibSp,Fare]]
    df = pd.DataFrame(data, columns = ['Sex', 'Pclass','Age', 'Parch','SibSp','Fare'])
    X1 = datasets(df)
    print(X1)
    yy = model(X1)
    if yy==True:
        print("Dead")
    else:
        print("Survived")
#############################################################################################################################
input_user()
