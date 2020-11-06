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
import pymysql

db_connection_str = 'mysql+pymysql://root:test@localhost/dissertation'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM final_data_5parks_1hour', con=db_connection)

##df['occupancy'].describe()

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

group_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
print(pd.cut(df['occupancy'], bins, labels=group_names).value_counts())
df['class'] = pd.cut(df['occupancy'], bins, labels=group_names)

print(df['class'])






