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

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


db_connection_str = 'mysql+pymysql://root:Fd=D((:+%"`2fBRt@localhost/dissertation'
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
# TRAINING THE DECSISION TREE CLASSIFICATION MODEL ON THE TRAINING SET
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
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
[[   0    1   18   41   58   89  105  110   56   49]
 [   7   14   43  114  199  296  317  303  200  166]
 [  11   57  151  343  610  860  980  842  627  560]
 [  21  114  375  798 1497 2137 2362 2188 1466 1271]
 [  57  222  632 1491 2688 3933 4424 4034 2815 2359]
 [  83  316  886 2088 3936 5775 6279 5616 3909 3362]
 [  83  351 1016 2347 4356 6254 7132 6337 4449 3819]
 [ 100  285  892 2091 3917 5516 6411 5655 3933 3392]
 [  61  228  638 1424 2783 3811 4375 3812 2767 2347]
 [  55  184  493 1202 2382 3411 3846 3360 2364 2081]]
0.14681452466078201
******* classification_report *********
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       527
           1       0.01      0.01      0.01      1659
           2       0.03      0.03      0.03      5041
           3       0.07      0.07      0.07     12229
           4       0.12      0.12      0.12     22655
           5       0.18      0.18      0.18     32250
           6       0.20      0.20      0.20     36144
           7       0.18      0.18      0.18     32192
           8       0.12      0.12      0.12     22246
           9       0.11      0.11      0.11     19378

    accuracy                           0.15    184321
   macro avg       0.10      0.10      0.10    184321
weighted avg       0.15      0.15      0.15    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.14419790592958281, 0.14718168502142895, 0.1442599826388889, 0.13981119791666666, 0.08881293402777778, 0.11897786458333333, 0.09016927083333333, 0.11214192708333333, 0.13834635416666666, 0.14708116319444445]
Accuracy: 12.71 %
Accuracy: 0.1271 (0.0219)

'''















