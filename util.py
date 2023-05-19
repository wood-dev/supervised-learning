import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

GRAPH_FOLDER = './graph'

def getFullFilePath(filename):
    return os.path.join(GRAPH_FOLDER, filename)

def splitData(data, train_size, normalize = False):
    X = data.iloc[:,:-1]
    Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=train_size/100)

    if normalize:
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    return X_train, X_test, y_train, y_test