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

db_connection_str = 'mysql+pymysql://root:Fd=D((:+%"`2fBRt@localhost/dissertation'
db_connection = create_engine(db_connection_str)

accepted_columns = ['parkName', 'day', 'month', 'year', 'hour', 'weekDay', 'occupancy', 'isWeekend', 'isHoliday',
                    'precipIntensity',
                    'precipProbability', 'temperature', 'apparentTemperature',
                    'windSpeed', 'place1Type', 'place2Type', 'place3Type', 'place4Type', 'place5Type',
                    'parkingGroupName', 'numberOfSpaces', 'hourlyPrice']

###accepted_columns = ['parkName', 'occupancy', 'parkingGroupName']

drop_cols = []

dataset = pd.read_sql('SELECT * FROM final_data_10parks_1hour', con=db_connection)

for col in dataset.columns.tolist():
    if col not in accepted_columns:
        drop_cols.append(col)

## REMOVE UNUSED COLS
dataset.drop(columns=drop_cols, inplace=True, axis=1)

'''bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
group_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
print(pd.cut(dataset['occupancy'], bins, labels=group_names).value_counts())
dataset['class'] = pd.cut(dataset['occupancy'], bins, labels=group_names)'''

##dataset.insert(0, 'class', pd.cut(dataset['occupancy'], bins, labels=group_names))


## SET CLASSES ACCORDING OCCUPANCY
dataset['class'] = pd.cut(dataset['occupancy'], bins=10)
##print(dataset['class'].unique())

'''
GENERATED CLASSES
    0: (-0.1, 10]
    1: (10, 20]
    2: (20, 30]
    3: (30, 40]
    4: (40, 50]
    5: (50, 60]
    6: (60, 70]
    7: (70, 80]
    8: (80, 90]
    9: (90, 100]
'''

dataset.drop(columns=['occupancy'], inplace=True, axis=1)

## ENCODING CATEGORICAL DATA WITH DUMMY APPROACH
dataset = pd.get_dummies(dataset, columns=['parkName', 'parkingGroupName', 'place1Type', 'place2Type', 'place3Type',
                                           'place4Type', 'place5Type'], drop_first=True)

## ENCODE TARGET VARIABLE WITH LABEL ENCODER
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
dataset['class_code'] = lb_make.fit_transform(dataset['class'])

##print(dataset[["class", "class_code"]].head(50))
# DROP COLUMN CLASS
dataset.drop(columns=['class'], inplace=True, axis=1)

print(list(dataset.columns))

##TAKING CARE OF MISSING DATA
'''
    THE MISSING DATA IN precipIntensity and precipProbability is on summer, between 19 May 2018 - 29 August 2018
    So nan values assume previous val
    THE IMPACT ON MEAN WAS TESTED AND ITS MINIMUM
    #print(dataset['precipIntensity'].mean())
    #print(dataset['precipProbability'].mean())
    FOR temperature, apparentTemperature and windSpeed the used approach was the same because weather conditions are stable

'''

dataset['precipIntensity'].fillna(method='pad', inplace=True)
dataset['precipProbability'].fillna(method='pad', inplace=True)
dataset['temperature'].fillna(method='pad', inplace=True)
dataset['apparentTemperature'].fillna(method='pad', inplace=True)
dataset['windSpeed'].fillna(method='pad', inplace=True)

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
# TRAINING THE KNN MODEL ON THE TRAINING SET
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)


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
[[   0    7   25   62   84   86  105   82   47   29]
 [   8   32   73  140  246  347  347  249  122   95]
 [  31   86  214  419  721 1041 1119  770  366  274]
 [  66  210  548 1045 1800 2470 2648 1867  846  729]
 [ 119  360  985 2008 3375 4622 4849 3404 1727 1206]
 [ 148  496 1484 2805 4728 6538 6853 5008 2439 1751]
 [ 154  564 1620 3265 5358 7396 7710 5493 2648 1936]
 [ 158  556 1459 2828 4656 6595 6835 4876 2404 1825]
 [ 122  367 1022 1981 3369 4387 4736 3358 1698 1206]
 [  90  286  833 1683 2988 3837 4225 2951 1398 1087]]
0.14417782021581915
******* classification_report *********
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       527
           1       0.01      0.02      0.01      1659
           2       0.03      0.04      0.03      5041
           3       0.06      0.09      0.07     12229
           4       0.12      0.15      0.14     22655
           5       0.18      0.20      0.19     32250
           6       0.20      0.21      0.20     36144
           7       0.17      0.15      0.16     32192
           8       0.12      0.08      0.09     22246
           9       0.11      0.06      0.07     19378

    accuracy                           0.14    184321
   macro avg       0.10      0.10      0.10    184321
weighted avg       0.15      0.14      0.14    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.14870069983182335, 0.14799544295556882, 0.14615885416666666, 0.1332465277777778, 0.1396484375, 0.13368055555555555, 0.1364474826388889, 0.13780381944444445, 0.1393771701388889, 0.1403537326388889]
Accuracy: 14.03 %
Accuracy: 0.1403 (0.0053)

'''















