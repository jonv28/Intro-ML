import pandas as pd
import numpy as np 
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')

print("PRINTING RAW DATA")
print(data.head())
print("\n")
X = data[[
    'buying', 
    'maint',
    'safety'
]].values

y = data[['class']]

print('PRINTING DATA TO BE USED FOR ML')
print(X,y)
print('\n')

#Converting feautres to make it ML understnable 
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:,i] = Le.fit_transform(X[:,i])


print('PRINTING TRANSFORMED FEATURES')
print(X)
print('\n')


#Converting labels to make them ML understnadable 

label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)
print('PRINTING TRNASFORMED LABELS')
print(y)
print('\n')



#Creatting a model 
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)

predicitions = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test,predicitions)

print('PRINTING PREDICTIONS')
print(predicitions)
print('\n')

print('PRINTING ACCURARACY')
print(accuracy)
print('\n')

print('PRINTING MANUAL TEST')
print('actual value', y[20])
print('predicted value', knn.predict(X)[20])


