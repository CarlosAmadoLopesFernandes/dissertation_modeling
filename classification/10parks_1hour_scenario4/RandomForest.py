
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


db_connection_str = 'mysql+pymysql://root:Fd=D((:+%"`2fBRt@localhost/dissertation'
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
# TRAINING THE RANDOM FOREST CLASSIFICATION MODEL ON THE TRAINING SET
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
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
[[   0    3   13   35   77   94  101  100   54   50]
 [   4   18   43  102  207  307  327  291  197  163]
 [  19   58  143  330  596  823 1020  895  616  541]
 [  32  132  385  820 1486 2114 2486 2115 1447 1212]
 [  66  210  620 1464 2740 4073 4469 3922 2760 2331]
 [  85  322  933 2075 3923 5811 6327 5622 3915 3237]
 [ 113  353 1019 2340 4370 6454 7193 6184 4389 3729]
 [  82  305  904 2157 3777 5676 6335 5742 3872 3342]
 [  57  202  664 1377 2727 3808 4542 3922 2728 2219]
 [  49  179  523 1261 2378 3440 3806 3384 2335 2023]]
0.14766629955349633
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
           9       0.11      0.10      0.11     19378

    accuracy                           0.15    184321
   macro avg       0.10      0.10      0.10    184321
weighted avg       0.15      0.15      0.15    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.13871860250637444, 0.14528291650843594, 0.1474066840277778, 0.15272352430555555, 0.1395941840277778, 0.14127604166666666, 0.13427734375, 0.15223524305555555, 0.1465386284722222, 0.14784071180555555]
Accuracy: 14.46 %
Accuracy: 0.1446 (0.0057)

'''















