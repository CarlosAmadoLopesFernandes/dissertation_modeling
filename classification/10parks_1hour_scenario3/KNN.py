import pandas as pd
import numpy as np
from sqlalchemy import create_engine


db_connection_str = 'mysql+pymysql://root:test@localhost/dissertation'
db_connection = create_engine(db_connection_str)

accepted_columns = ['parkName', 'day', 'month', 'year', 'hour', 'weekDay', 'occupancy', 'isWeekend', 'isHoliday',
                    'place1Type', 'place2Type', 'place3Type', 'place4Type', 'place5Type',
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
[[   3    7   24   56   82  106  114   72   34   29]
 [  11   33   66  156  247  326  357  252  124   87]
 [  18   96  205  450  704 1008 1096  813  368  283]
 [  59  209  539 1046 1890 2466 2629 1903  834  654]
 [  97  359  995 2004 3398 4707 4818 3462 1684 1131]
 [ 147  520 1483 2796 4855 6491 6888 4957 2396 1717]
 [ 156  597 1627 3195 5250 7380 7773 5438 2764 1964]
 [ 139  502 1391 2860 4739 6655 6743 4988 2399 1776]
 [ 118  348 1007 1936 3326 4438 4809 3415 1685 1164]
 [  90  291  875 1653 2979 3874 4255 2898 1405 1058]]
0.14474747858355802
******* classification_report *********
              precision    recall  f1-score   support

           0       0.00      0.01      0.00       527
           1       0.01      0.02      0.01      1659
           2       0.02      0.04      0.03      5041
           3       0.06      0.09      0.07     12229
           4       0.12      0.15      0.14     22655
           5       0.17      0.20      0.19     32250
           6       0.20      0.22      0.21     36144
           7       0.18      0.15      0.17     32192
           8       0.12      0.08      0.09     22246
           9       0.11      0.05      0.07     19378

    accuracy                           0.14    184321
   macro avg       0.10      0.10      0.10    184321
weighted avg       0.15      0.14      0.14    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.14647642814517442, 0.14365540064015625, 0.14925130208333334, 0.1359592013888889, 0.13536241319444445, 0.13444010416666666, 0.1325412326388889, 0.13047960069444445, 0.1312934027777778, 0.13802083333333334]
Accuracy: 13.77 %
Accuracy: 0.1377 (0.0062)

Process finished with exit code 0


'''















