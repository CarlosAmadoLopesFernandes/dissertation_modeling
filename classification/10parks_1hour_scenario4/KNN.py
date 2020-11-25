'''

    MULTI-CLASS CLASSIFICATION
    Refers to those classification tasks that have more than two class labels.

    Unlike binary classification, multi-class classification does not have the notion of normal and abnormal outcomes.
    Instead, examples are classified as belonging to one among a range of known classes

    Popular algorithms that can be used for mult-class classification include:
        . KNN
        . Decision trees
        . Naive Bayes
        . Random Forest
        . Gradient Boosting
'''
import mysql.connector
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import pymysql

db_connection_str = 'mysql+pymysql://root:test@localhost/dissertation'
db_connection = create_engine(db_connection_str)

accepted_columns = ['parkName', 'day', 'month', 'year', 'hour', 'weekDay', 'occupancy', 'isWeekend', 'isHoliday',
                    'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature',
                    'dewPoint', 'humidity', 'pressure', 'windSpeed', 'windGust', 'windBearing', 'cloudCover',
                    'uvIndex', 'visibility', 'place1Type', 'place2Type', 'place3Type', 'place4Type', 'place5Type']


drop_cols = []

dataset = pd.read_sql('SELECT * FROM final_data_10parks_1hour', con=db_connection)

for col in dataset.columns.tolist():
    if col not in accepted_columns:
        drop_cols.append(col)

## REMOVE UNUSED COLS
dataset.drop(columns=drop_cols, inplace=True, axis=1)

## SET CLASSES ACCORDING OCCUPANCY
dataset['class'] = pd.cut(dataset['occupancy'], bins=10)


dataset.drop(columns=['occupancy'], inplace=True, axis=1)

## ENCODING CATEGORICAL DATA WITH DUMMY APPROACH
dataset = pd.get_dummies(dataset, columns=['parkName', 'place1Type', 'place2Type', 'place3Type',
                                           'place4Type', 'place5Type'], drop_first=True)

## ENCODE TARGET VARIABLE WITH LABEL ENCODER
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
dataset['class_code'] = lb_make.fit_transform(dataset['class'])


# DROP COLUMN CLASS
dataset.drop(columns=['class'], inplace=True, axis=1)

print(list(dataset.columns))

##TAKING CARE OF MISSING DATA
dataset['precipIntensity'].fillna(method='pad', inplace=True)
dataset['precipProbability'].fillna(method='pad', inplace=True)
dataset['temperature'].fillna(method='pad', inplace=True)
dataset['apparentTemperature'].fillna(method='pad', inplace=True)
dataset['windSpeed'].fillna(method='pad', inplace=True)
dataset['humidity'].fillna(method='pad', inplace=True)
dataset['pressure'].fillna(method='pad', inplace=True)
dataset['windGust'].fillna(method='pad', inplace=True)
dataset['windBearing'].fillna(method='pad', inplace=True)
dataset['cloudCover'].fillna(method='pad', inplace=True)
dataset['uvIndex'].fillna(method='pad', inplace=True)
dataset['visibility'].fillna(method='pad', inplace=True)


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("split dataset")
## SPLITIING INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


print("training model")
# TRAINING THE KNN MODEL ON THE TRAINING SET
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)



print("predicting result")
## PREDICTING THE TEST SET RESULT
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print("MAKING CONFUSION MATRIX")
##MAKING THE CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print("DONE ***********")

'''

MAKING CONFUSION MATRIX
[[   2    4   10   18   35   43   41   35   17   18]
 [   3   17   41   87  111  172  169  139   53   47]
 [  18   43  104  251  396  515  568  385  177  128]
 [  32  107  284  529  943 1220 1291  959  435  333]
 [  59  200  505  991 1619 2376 2381 1719  772  614]
 [  75  254  675 1412 2398 3327 3537 2492 1215  875]
 [  98  325  840 1662 2772 3628 3773 2681 1297  933]
 [  83  252  739 1455 2367 3227 3338 2523 1116  872]
 [  47  160  530 1007 1679 2220 2317 1745  815  596]
 [  47  155  445  899 1511 1991 2034 1513  714  479]]
0.14309740562710907
'''















