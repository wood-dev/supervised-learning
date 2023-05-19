import csv
import numpy as np
import pandas as pd
import os
from DecisionTree import DecisionTree
from NeuralNetworks import NeuralNetworks
from KNearestNeighbors import KNearestNeighbors
from Boosting import Boosting
from SupportVectorMachine import SupportVectorMachine
from sklearn import tree
from sklearn.impute import SimpleImputer

DATA_FOLDER = './data'

FILENAME_1 = 'online_shoppers_intention.csv'
CATEGORICAL_COLUMNS_1 = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
Y_COLUMN_1 = 'Revenue'
IDENTIFIER_1 = 1

FILENAME_2 = 'census_income.csv'
CATEGORICAL_COLUMNS_2 = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
Y_COLUMN_2 = 'greater-than-50k'
IDENTIFIER_2 = 2

NUMERIC_COLUMNS = []

def loadData_1(encode_category = False):
    fullFilename = os.path.join(DATA_FOLDER, FILENAME_1)
    df = pd.read_csv(fullFilename)
    df.head()

    global NUMERIC_COLUMNS
    NUMERIC_COLUMNS = df.columns.difference(CATEGORICAL_COLUMNS_1)
    NUMERIC_COLUMNS = NUMERIC_COLUMNS.drop(Y_COLUMN_1)

    if encode_category:
        df_oneHot = df[CATEGORICAL_COLUMNS_1]
        df_oneHot = pd.get_dummies(df_oneHot, drop_first=True)
        df_droppedOneHot = df.drop(CATEGORICAL_COLUMNS_1, axis=1)
        return pd.concat([df_oneHot, df_droppedOneHot], axis=1)
    else:
        return df


def loadData_2(encode_category = False):
    fullFilename = os.path.join(DATA_FOLDER, FILENAME_2)
    df = pd.read_csv(fullFilename)
    df.head()

    # categorical value
    df[CATEGORICAL_COLUMNS_2].fillna('Nan')
    # numeric value
    global NUMERIC_COLUMNS
    NUMERIC_COLUMNS = df.columns.difference(CATEGORICAL_COLUMNS_2)
    NUMERIC_COLUMNS = NUMERIC_COLUMNS.drop(Y_COLUMN_2)

    df_numeric = df[NUMERIC_COLUMNS]
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    df_numeric = imputer.fit_transform(df_numeric)

    if encode_category:
        df_oneHot = df[CATEGORICAL_COLUMNS_2]
        df_oneHot = pd.get_dummies(df_oneHot, drop_first=True)
        df_droppedOneHot = df.drop(CATEGORICAL_COLUMNS_2, axis=1)
        return pd.concat([df_oneHot, df_droppedOneHot], axis=1)
    else:
        return df

def main():

    data = loadData_1(encode_category = True)
    dt1 = DecisionTree(IDENTIFIER_1, FILENAME_1)
    dt1.analyze(data)

    data = loadData_2(encode_category = True)
    dt2 = DecisionTree(IDENTIFIER_2, FILENAME_2)
    dt2.analyze(data)

    # data = loadData_1(encode_category = True)
    # nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1)
    # nn.analyze(data)

    # data = loadData_2(encode_category = True)
    # nn = NeuralNetworks(IDENTIFIER_2, FILENAME_2)
    # nn.analyze(data)

    # data = loadData_1(encode_category = True)
    # knn = KNearestNeighbors(IDENTIFIER_1, FILENAME_1)
    # knn.analyze(data)

    # data = loadData_2(encode_category = True)
    # knn = KNearestNeighbors(IDENTIFIER_2, FILENAME_2)
    # knn.analyze(data)

    # data = loadData_1(encode_category = True)
    # bt = Boosting(IDENTIFIER_1, FILENAME_1)
    # bt.analyze(data)

    # data = loadData_2(encode_category = True)
    # bt = Boosting(IDENTIFIER_2, FILENAME_2)
    # bt.analyze(data)

    # data = loadData_1(encode_category = True)
    # svm = SupportVectorMachine(IDENTIFIER_1, FILENAME_1)
    # svm.analyze(data)

    # data = loadData_2(encode_category = True)
    # svm = SupportVectorMachine(IDENTIFIER_2, FILENAME_2)
    # svm.analyze(data)


if __name__ == "__main__":
    main()