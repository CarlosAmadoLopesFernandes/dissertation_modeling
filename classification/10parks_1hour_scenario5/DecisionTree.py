import pandas as pd
import numpy as np
from sqlalchemy import create_engine

db_connection_str = 'mysql+pymysql://root:test@localhost/dissertation'
db_connection = create_engine(db_connection_str)

accepted_columns = ['parkName', 'day', 'month', 'year', 'hour', 'weekDay', 'occupancy', 'isWeekend', 'isHoliday',
                    'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature',
                    'dewPoint', 'humidity', 'pressure', 'windSpeed', 'windGust', 'windBearing', 'cloudCover',
                    'uvIndex', 'visibility',
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
dataset = pd.get_dummies(dataset, columns=['parkName', 'parkingGroupName'], drop_first=True)

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
[[   0    1   18   33   69   90  116  100   48   52]
 [   7   13   36  115  197  299  320  302  188  182]
 [  14   46  138  324  597  884 1016  844  619  559]
 [  36  133  352  785 1485 2127 2452 2160 1438 1261]
 [  60  224  662 1460 2649 3977 4462 3994 2795 2372]
 [  77  313  910 2088 3910 5764 6332 5642 3866 3348]
 [ 101  349  999 2346 4252 6319 7188 6380 4350 3860]
 [  76  302  897 2120 3863 5668 6405 5627 3829 3405]
 [  59  220  631 1423 2701 3749 4408 3889 2776 2390]
 [  45  172  534 1249 2360 3458 3809 3398 2318 2035]]
0.14634794733101492
******* classification_report *********
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       527
           1       0.01      0.01      0.01      1659
           2       0.03      0.03      0.03      5041
           3       0.07      0.06      0.06     12229
           4       0.12      0.12      0.12     22655
           5       0.18      0.18      0.18     32250
           6       0.20      0.20      0.20     36144
           7       0.17      0.17      0.17     32192
           8       0.12      0.12      0.12     22246
           9       0.10      0.11      0.10     19378

    accuracy                           0.15    184321
   macro avg       0.10      0.10      0.10    184321
weighted avg       0.15      0.15      0.15    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.14311289535072966, 0.14658492920305974, 0.1448025173611111, 0.14208984375, 0.09130859375, 0.11707899305555555, 0.0886501736111111, 0.1091579861111111, 0.1392144097222222, 0.14561631944444445]
Accuracy: 12.68 %
Accuracy: 0.1268 (0.0220)

Process finished with exit code 0



'''















