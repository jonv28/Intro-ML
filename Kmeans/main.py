from operator import mod
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd 

bc = load_breast_cancer()

X = scale(bc.data) 
print(X)

y = bc.target
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#random state means the randomization wont change basically it's a seed 
model = KMeans(n_clusters=2, random_state=0)

model.fit(X_train)

predictions = model.predict(X_test)
labels = model.labels_

print('Labels', labels)
print("Predictions", predictions)
print('accuracry', accuracy_score(y_test, predictions))
print('Actual', y_test)

#Sometimes cluster 1 can be for label 0 making ur accuracy garbage 
#This is because we didn;t give it the labels 
#This is a way to check to see if labels are right
print(pd.crosstab(y_train, labels))

#another way to check accuracy 
