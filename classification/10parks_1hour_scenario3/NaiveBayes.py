import pandas as pd
import numpy as np
from sqlalchemy import create_engine


db_connection_str = 'mysql+pymysql://root:Fd=D((:+%"`2fBRt@localhost/dissertation'
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
# TRAINING THE NAIVE BAYES MODEL ON THE TRAINING SET
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
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
[[    0     0     0     0    78     0   387    60     2     0]
 [    0     0     0     0   225     3  1217   203    11     0]
 [    0     0     0     0   743     5  3687   586    20     0]
 [    0     0     0     0  1782    11  8948  1427    61     0]
 [    0     0     0     0  3188    26 16633  2729    79     0]
 [    0     0     0     0  4668    41 23590  3823   128     0]
 [    0     0     0     0  5233    40 26475  4259   137     0]
 [    0     0     0     0  4695    35 23464  3879   119     0]
 [    0     0     0     0  3218    30 16226  2680    92     0]
 [    0     0     0     0  2833    23 14209  2226    87     0]]
0.1826975765105441

******* classification_report *********
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       527
           1       0.00      0.00      0.00      1659
           2       0.00      0.00      0.00      5041
           3       0.00      0.00      0.00     12229
           4       0.12      0.14      0.13     22655
           5       0.19      0.00      0.00     32250
           6       0.20      0.73      0.31     36144
           7       0.18      0.12      0.14     32192
           8       0.12      0.00      0.01     22246
           9       0.00      0.00      0.00     19378

    accuracy                           0.18    184321
   macro avg       0.08      0.10      0.06    184321
weighted avg       0.13      0.18      0.10    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.19638691477241904, 0.19638691477241904, 0.19639756944444445, 0.17616102430555555, 0.1784939236111111, 0.018663194444444444, 0.03683810763888889, 0.1782769097222222, 0.1796875, 0.17887369791666666]
Accuracy: 15.36 %
Accuracy: 0.1536 (0.0636)

Process finished with exit code 0


'''















