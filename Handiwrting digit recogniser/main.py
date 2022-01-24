
from PIL import Image
import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist

x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()

#This is to reshape the data so it can be used for us, -1 means don't change number
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1,28*28))

#pixels range from 0-255 neural networks work better for 0 to 1 so we need to scale down our data set
x_train = (x_train/256)
x_test = (x_test/256)

#Data is now ready so we can make our model 
clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64)) 
#adam works good for large amouts of data

clf.fit(x_train, y_train)

prediction = clf.predict(x_test)
acc = confusion_matrix(y_test , prediction)

#do trace of matrix / all elements in the matrix summed up 

def accuracy(cm):
    diagonal = cm.trace()
    elements = cm.sum()
    return diagonal/elements

print(accuracy(acc))


#test on Number made by me
img = Image.open('Five.png')
data = list(img.getdata())

#data is the opposite of mnist data set so we need to inverse the data 
for i in range(len(data)):
    data[i] = 255 - data[i]
Five = np.array(data)/256

p = clf.predict([Five])
print(p)
