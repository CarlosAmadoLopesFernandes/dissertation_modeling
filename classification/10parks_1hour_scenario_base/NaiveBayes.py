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
[[    0     0     0     0    22   302   153    34    14     2]
 [    0     0     0     0    54   987   487    96    35     0]
 [    0     0     0     0   197  2909  1564   269   101     1]
 [    0     0     0     0   448  7097  3741   665   271     7]
 [    0     0     0     0   833 13154  6990  1154   516     8]
 [    0     0     0     0  1159 18576 10051  1688   768     8]
 [    0     0     0     0  1347 20954 11221  1818   788    16]
 [    0     0     0     0  1142 18617 10013  1703   700    17]
 [    0     0     0     0   835 12735  6999  1182   488     7]
 [    0     0     0     0   768 11141  6023   965   475     6]]
0.17809690702632908

******* classification_report *********
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       527
           1       0.00      0.00      0.00      1659
           2       0.00      0.00      0.00      5041
           3       0.00      0.00      0.00     12229
           4       0.12      0.04      0.06     22655
           5       0.17      0.58      0.27     32250
           6       0.20      0.31      0.24     36144
           7       0.18      0.05      0.08     32192
           8       0.12      0.02      0.04     22246
           9       0.08      0.00      0.00     19378

    accuracy                           0.18    184321
   macro avg       0.09      0.10      0.07    184321
weighted avg       0.14      0.18      0.12    184321

******  ACCURACY WITH CROSS VALIDATION ****** 
[0.19416264308577008, 0.19524765366462324, 0.17496744791666666, 0.17447916666666666, 0.17952473958333334, 0.19325086805555555, 0.0920138888888889, 0.17415364583333334, 0.17637803819444445, 0.1732855902777778]
Accuracy: 17.27 %
Accuracy: 0.1727 (0.0282)


'''

import pickle
pickle.dump(classifier, open('../../api/models/10_parks_1_hour_scenario1/naive_bayes.pkl', 'wb'))
pickle.dump(list(dataset.columns)[:-1], open('../../api/models/10_parks_1_hour_scenario1/model_columns.pkl', 'wb'))
