import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('float_format', '{:f}'.format)

sns.set(style="white", color_codes=True)

def readCSV(fileName):
    df = pd.read_csv('data\\' + fileName,encoding='utf-8')
    df = df.rename(columns=lambda x: x.strip())
    df = df.rename(columns=lambda x: x.replace(' ','_'))
    return df

def writeCSV(df, fileName):
    df.to_csv('data\\' + fileName, index=False, header=True)
    return df

def splitData(df,trainFileName ,testFileName):
    df.launched = df.launched.astype('datetime64')
    trainData = df[df.launched.dt.year <=2015]
    testData = df[df.launched.dt.year > 2015]
    trainData = writeCSV(trainData,trainFileName)
    testData = writeCSV(testData,testFileName)
    return trainData


def getDescribe(data):
    describeDF = data.describe(include='all')
    describeDF = describeDF.reset_index()
    describeDF = describeDF.rename({'index': 'attribute'}, axis='columns')

    nullValues = {}
    nullValues['attribute'] = 'null_values'
    for col in data.columns:
        nullValues[col] = np.sum(data[col].isnull())
    describeDF = describeDF.append(nullValues, ignore_index=True)

    notNullValues = {}
    notNullValues['attribute'] = 'not_null_values'
    for col in data.columns:
        notNullValues[col] = np.sum(data[col].notnull())
    describeDF = describeDF.append(notNullValues, ignore_index=True)

    types = {}
    types['attribute'] = 'dtype'
    for col in data.columns:
        types[col] = data.dtypes[col]
    describeDF = describeDF.append(types, ignore_index=True)

    return describeDF


def getNumericData(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return df.select_dtypes(include=numerics)


def getScalingData(df):
    lables = getNumericData(df).columns
    scaleData = df.copy()
    scaleData[lables] = MinMaxScaler().fit_transform(scaleData[lables])
    return scaleData


## PLOTING FUNCTIONS

def plotCorr(data):
    corr = data.corr()
    return corr.style.background_gradient(cmap='GnBu').set_precision(2)

