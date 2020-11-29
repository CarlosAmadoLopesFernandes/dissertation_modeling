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
                    'uvIndex', 'visibility', 'place1Type', 'place2Type', 'place3Type', 'place4Type', 'place5Type',
                    'parkingGroupName', 'numberOfSpaces', 'handicappedPlaces', 'hourlyPrice',
                    'has_car_washing', 'has_bike_sharing', 'has_bike_parking', 'has_elevator', 'has_information_point', 'has_toilet',
                    'has_petrol_station', 'has_shop', 'has_restaurant', 'has_electric_charging_station', 'has_overnight_accommodation',
                    'has_kiosk', 'has_pharmacy', 'has_cafe', 'has_medical_facility', 'has_vending_machine', 'has_spare_parts_shopping']


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
dataset = pd.get_dummies(dataset, columns=['parkName', 'parkingGroupName', 'place1Type', 'place2Type', 'place3Type',
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

## WHEN IS NULL, REPLACE BY ZERO
dataset['handicappedPlaces'].fillna(0, inplace=True)


##print(dataset.isnull().sum())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("split dataset")
## SPLITIING INTO TRAINING SET AND TEST SET
##from sklearn.model_selection import train_test_split

##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    print("TRAIN: ", train_index, "TEST: ", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


print("training model")
# TRAINING THE GRADIENT BOOST CLASSIFICATION MODEL ON THE TRAINING SET
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators=10, criterion='friedman_mse', loss='deviance', learning_rate=0.1, random_state=0)
classifier.fit(X_train, y_train)


# PREDICTING A NEW RESULT
##print(classifier.predict(sc.transform([[30, 87000]])))

print("predicting result")
## PREDICTING THE TEST SET RESULT
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

## METRICS
print("MAKING CONFUSION MATRIX")
##MAKING THE CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

report = classification_report(y_test, y_pred)
print("******* classification_report *********")
print(report)


print("******  ACCURACY WITH CROSS VALIDATION ****** ")
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(list(accuracies))
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Accuracy: %.4f (%.4f)" % (accuracies.mean(), accuracies.std()))

'''
MAKING CONFUSION MATRIX
[[    0     0     0     0     1     0   526     0     0     0]
 [    0     0     1     0     0     0  1655     3     0     0]
 [    0     0     0     0     1    12  5024     4     0     0]
 [    1     1     1     0     6    17 12188    11     1     3]
 [    0     0     0     0     5    39 22593    16     1     1]
 [    1     1     1     1     3    62 32149    25     4     3]
 [    1     3     1     1    11    59 36030    32     4     2]
 [    0     6     1     0     1    58 32095    29     1     1]
 [    1     1     0     0     4    44 22165    24     4     3]
 [    1     2     1     0     5    41 19308    16     3     1]]
0.19602215699784614
******* classification_report *********
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       527
           1       0.00      0.00      0.00      1659
           2       0.00      0.00      0.00      5041
           3       0.00      0.00      0.00     12229
           4       0.14      0.00      0.00     22655
           5       0.19      0.00      0.00     32250
           6       0.20      1.00      0.33     36144
           7       0.18      0.00      0.00     32192
           8       0.22      0.00      0.00     22246
           9       0.07      0.00      0.00     19378

    accuracy                           0.20    184321
   macro avg       0.10      0.10      0.03    184321
weighted avg       0.15      0.20      0.07    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.19633266424347637, 0.1964411653013617, 0.18343098958333334, 0.1960177951388889, 0.08365885416666667, 0.1960720486111111, 0.1882052951388889, 0.19249131944444445, 0.1967230902777778, 0.19596354166666666]
Accuracy: 18.25 %
Accuracy: 0.1825 (0.0332)

Process finished with exit code 0



'''















