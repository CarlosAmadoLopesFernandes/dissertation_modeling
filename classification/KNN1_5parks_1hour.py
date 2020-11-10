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
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import pymysql

db_connection_str = 'mysql+pymysql://root:test@localhost/dissertation'
db_connection = create_engine(db_connection_str)

accepted_columns = ['parkName', 'date', 'occupancy', 'isWeekend', 'isHoliday', 'precipIntensity',
                    'precipProbability', 'temperature', 'apparentTemperature',
                    'windSpeed', 'place1Type', 'place2Type', 'place3Type', 'place4Type', 'place5Type',
                    'parkingGroupName', 'numberOfSpaces', 'hourlyPrice']

drop_cols = []

dataset = pd.read_sql('SELECT * FROM final_data_5parks_1hour', con=db_connection)


for col in dataset.columns.tolist():
    if col not in accepted_columns:
        drop_cols.append(col)


## REMOVE UNUSED COLS
dataset.drop(columns=drop_cols, inplace=True, axis=1)


##dataset['occupancy'].describe()

## SET CLASSES ACCORDING OCCUPANCY
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
group_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
##print(pd.cut(dataset['occupancy'], bins, labels=group_names).value_counts())

dataset['class'] = pd.cut(dataset['occupancy'], bins, labels=group_names)
##dataset.insert(0, 'class', pd.cut(dataset['occupancy'], bins, labels=group_names))

dataset.drop(columns=['occupancy'], inplace=True, axis=1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



## SPLITIING INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


print(X_train)













