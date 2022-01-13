from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

#Feautres/ labels
X = boston.data
y = boston.target



#algorithm
l_reg = linear_model.LinearRegression()

#Making some plots 
#The .T takes the transpose good to know
plt.scatter(X.T[5],y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#Train 
model = l_reg.fit(X_train, y_train)
predicitions = l_reg.predict(X_test)
print("Predictions: ", predicitions)
print("R^2 value", l_reg.score(X,y))
print("coedd:" , l_reg.coef_)

