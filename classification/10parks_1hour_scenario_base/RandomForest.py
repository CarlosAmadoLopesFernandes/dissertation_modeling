import pandas as pd
import numpy as np
from sqlalchemy import create_engine

db_connection_str = 'mysql+pymysql://root:test@localhost/dissertation'
db_connection = create_engine(db_connection_str)

accepted_columns = ['parkName', 'day', 'month', 'year', 'hour', 'weekDay', 'occupancy', 'isWeekend', 'isHoliday',
                    'precipIntensity',
                    'precipProbability', 'temperature', 'apparentTemperature',
                    'windSpeed', 'place1Type', 'place2Type', 'place3Type', 'place4Type', 'place5Type',
                    'parkingGroupName', 'numberOfSpaces', 'hourlyPrice']


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

##print(dataset[["class", "class_code"]].head(50))
# DROP COLUMN CLASS
dataset.drop(columns=['class'], inplace=True, axis=1)

print(list(dataset.columns))

##TAKING CARE OF MISSING DATA

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
# TRAINING THE RANDOM FOREST CLASSIFICATION MODEL ON THE TRAINING SET
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# PREDICTING A NEW RESULT
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
[[   1    0    3    6   16   63   55   42   13   24]
 [   1    2   10   27  110  167  220  165   80   57]
 [   2    8   26  108  238  564  710  539  237  153]
 [   9   17   65  238  645 1310 1676 1176  594  403]
 [   5   33  125  390 1219 2374 2949 2316 1007  818]
 [  11   44  174  545 1672 3500 4516 3254 1408 1136]
 [  13   49  171  598 1856 3831 4905 3702 1635 1249]
 [  13   44  174  579 1680 3365 4286 3205 1515 1111]
 [  14   40  120  377 1162 2322 3014 2326  992  749]
 [   9   34  109  369 1015 2028 2618 2011  885  710]]
0.16056683412723385

'''






