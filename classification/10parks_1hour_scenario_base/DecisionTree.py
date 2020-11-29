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
# TRAINING THE DECSISION TREE CLASSIFICATION MODEL ON THE TRAINING SET
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
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
[[   2    1   13   38   73   92   86  110   62   50]
 [   8   12   30  107  215  292  340  285  203  167]
 [  14   42  145  340  616  853  985  893  629  524]
 [  36  123  338  804 1492 2151 2433 2174 1414 1264]
 [  65  195  570 1490 2742 4090 4512 3984 2719 2288]
 [  73  298  927 2059 3975 5816 6365 5602 3818 3317]
 [  77  349 1016 2286 4414 6284 7276 6347 4290 3805]
 [  92  300  909 2098 3849 5545 6419 5793 3851 3336]
 [  59  199  613 1443 2715 3896 4423 3882 2673 2343]
 [  50  175  524 1229 2372 3380 3959 3357 2306 2026]]
0.1480514971164436

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
[0.14164813106927793, 0.11267834861389898, 0.06971571180555555, 0.1395941840277778, 0.10823567708333333, 0.08143446180555555, 0.1081271701388889, 0.10926649305555555, 0.11979166666666667, 0.1408420138888889]
Accuracy: 11.31 %
Accuracy: 0.1131 (0.0230)

'''

