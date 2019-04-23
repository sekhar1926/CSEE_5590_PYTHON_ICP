import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from scipy.misc import toimage
import tensorflow as tf
import numpy as np
# Load trained CNN model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
def show_imgs(X):
    plt.figure(1)
    k = 0
    for i in range(0,2):
        for j in range(0,2):
            plt.subplot2grid((2,2),(i,j))
            plt.imshow(toimage(X[k]))
            k = k+1
    indices=np.argmax(model.predict(X_test[:4]),1)
    plt.text(-20,50,[labels[x] for x in indices],horizontalalignment='center')



show_imgs(X_test[:4])

#Predicting the images using the model
class_pred = model.predict(X_test[:4], batch_size=32)
print(labels)
print("probabilities for each predicted class for first four images:\n",class_pred[:4])

#Getting the labels
labels_pred = np.argmax(class_pred,axis=1)
for i in range(4):
    j=labels_pred[i]
    print("Predicted class for "+str(i+1)+" image:",labels[j])

labels_pred1=labels_pred
y_test1=y_test[:4]
y_test1=y_test1.ravel()
for i in range(4):
    j=y_test1[i]
    print("Actual Class for "+str(i+1)+" image:",labels[j])

c=0
for i in range(4):
    if labels_pred1[i] == y_test1[i]:
        c=c+1

print("Number of correct predictions in first four images:",c)
# show the plot
plt.show()