import pandas as pd
import numpy as np
from sqlalchemy import create_engine

db_connection_str = 'mysql+pymysql://root:Fd=D((:+%"`2fBRt@localhost/dissertation'
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
[[   1    2   12   21   66   95  115  120   53   42]
 [   4    5   29   77  193  333  376  310  190  142]
 [  13   36  100  265  559  931 1191  981  527  438]
 [  20   80  235  598 1444 2326 2864 2393 1294  975]
 [  38  117  391 1105 2526 4468 5238 4368 2431 1973]
 [  43  184  613 1522 3763 6433 7345 6162 3455 2730]
 [  66  212  639 1747 4070 7151 8356 6961 3939 3003]
 [  58  175  553 1592 3612 6357 7322 6388 3439 2696]
 [  31  109  404 1039 2605 4262 5289 4239 2488 1780]
 [  29   93  358  897 2205 3774 4512 3771 2023 1716]]
0.15522376723216563
******* classification_report *********
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       527
           1       0.00      0.00      0.00      1659
           2       0.03      0.02      0.02      5041
           3       0.07      0.05      0.06     12229
           4       0.12      0.11      0.12     22655
           5       0.18      0.20      0.19     32250
           6       0.20      0.23      0.21     36144
           7       0.18      0.20      0.19     32192
           8       0.13      0.11      0.12     22246
           9       0.11      0.09      0.10     19378

    accuracy                           0.16    184321
   macro avg       0.10      0.10      0.10    184321
weighted avg       0.15      0.16      0.15    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.1719741767482233, 0.1638908479357674, 0.1257052951388889, 0.14735243055555555, 0.12890625, 0.125, 0.0812717013888889, 0.13319227430555555, 0.13785807291666666, 0.14708116319444445]
Accuracy: 13.62 %
Accuracy: 0.1362 (0.0237)

Process finished with exit code 0

'''


import pickle
pickle.dump(classifier, open('../../exported_models/10_parks_1_hour_scenario1/random_forest.pkl', 'wb'))
pickle.dump(list(dataset.columns)[:-1], open('../../exported_models/10_parks_1_hour_scenario1/model_columns.pkl', 'wb'))






