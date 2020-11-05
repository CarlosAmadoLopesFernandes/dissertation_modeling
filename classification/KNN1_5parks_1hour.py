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

#db_connection_str = 'mysql+pymysql://root:test@localhost/dissertation'
#db_connection = create_engine(db_connection_str)

#df = pd.read_sql('SELECT * FROM final_data_5parks_1hour', con=db_connection)

'''mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="test"
)

mycursor = mydb.cursor()
time_zone = pytz.timezone('Europe/Brussels')

query = "SELECT * FROM dissertation.final_data_5parks_1hour"
mycursor.execute(query)
all_data = mycursor.fetchall()'''



#print(df)

x = 2




