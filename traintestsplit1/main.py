from sklearn import datasets
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
#split it in feautres and lables 

X = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

print(X.shape)
print(y.shape)

#hours of study vs good/bad grades 
#10 diff studentds 
#train with 8 
#Predict with the remaning 2 
#will show the model accuracy 

#test size is perventage so 0.2 means 20 percent of data will be used for testing  
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#in general trainaing data should be much larger than testing data 

#making a model 
model = svm.SVC()
model.fit(X_train,y_train)

#Predicting
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

#printing results
print("Predictions: " ,predictions)
print("actual", y_test)
print("acuracy: " ,acc)


#Print out names
for i in range(len(predictions)):
    print(classes[predictions[i]])