#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:36:41 2024

@author: alejandrosierra
"""

#PLANTILLA  de Pre Procesado - Datos Faltantes

#Importar Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataSet 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Tratamiento de los NAs
from sklearn.impute import SimpleImputer
#X
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Dividir el data set en conjunto de entrenamiento y conjunto de testing 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 28)

