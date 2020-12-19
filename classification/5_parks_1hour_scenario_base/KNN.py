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

accepted_columns = ['parkName', 'day', 'month', 'year', 'hour', 'weekDay', 'occupancy', 'isWeekend', 'isHoliday', 'precipIntensity',
                  'precipProbability', 'temperature', 'apparentTemperature',
                    'windSpeed', 'place1Type', 'place2Type', 'place3Type', 'place4Type', 'place5Type',
                    'parkingGroupName', 'numberOfSpaces', 'hourlyPrice']

###accepted_columns = ['parkName', 'occupancy', 'parkingGroupName']

drop_cols = []

dataset = pd.read_sql('SELECT * FROM final_data_5parks_1hour', con=db_connection)




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
dataset = pd.get_dummies(dataset, columns=['parkName', 'parkingGroupName', 'place1Type', 'place2Type', 'place3Type', 'place4Type','place5Type'], drop_first=True)


## ENCODE TARGET VARIABLE WITH LABEL ENCODER
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
dataset['class_code'] = lb_make.fit_transform(dataset['class'])


##print(dataset[["class", "class_code"]].head(50))
#DROP COLUMN CLASS
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



print("training model")
#TRAINING THE KNN MODEL ON THE TRAINING SET
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

#PREDICTING A NEW RESULT
##print(classifier.predict(sc.transform([[30, 87000]])))

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
[[   2    3    6   15   30   34   31   21   10    7]
 [   3    7   26   51   49   89  107   78   25   27]
 [   4   21   68  127  225  254  324  193  112   79]
 [  18   63  142  278  455  632  671  536  262  169]
 [  31   92  268  544  946 1270 1244  887  474  351]
 [  40  125  395  799 1247 1830 1910 1364  679  453]
 [  29  130  427  906 1459 2017 2121 1472  729  532]
 [  33  145  386  827 1254 1845 1870 1349  617  465]
 [  25   97  261  551  827 1251 1268  938  447  349]
 [  35   79  248  501  797 1063 1117  831  417  269]]
0.14579472771833343
DONE ***********

Process finished with exit code 0

'''















