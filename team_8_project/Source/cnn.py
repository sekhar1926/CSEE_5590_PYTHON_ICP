
'''
!tar -xvf crowdai_train_2.tar

!tar -xvf crowdai_test.tar'''

import numpy as np
import keras
from keras.preprocessing import image
from keras.layers import MaxPooling2D,Convolution2D,Dropout, Flatten,Dense,Activation
from keras.models import Sequential,save_model
from keras.utils import np_utils
import os
import cv2
from sklearn.utils import shuffle

path = '/Users/adisekharrajuuppalapati/Downloads/crowdai'

leaf = os.listdir(path)
print(len(leaf),type(leaf))

print(leaf)

print(leaf[0][2:4])

x,y = [], []

for i in leaf:
    images = os.listdir(path+"/"+i)
    for j in images:
        img_path = path+"/"+i+"/"+j
        #Better method then cv2.imread
        img = image.load_img(img_path, target_size=(28,28))
        img = image.img_to_array(img)
        #print(img.shape)
        #img =img.flatten()
        #img = img.reshape(1,784)
#         print(img.shape)
#         img = img.reshape((28,28))
        img = img/255.0
        x.append(img)
        y.append(int(i[2:4]))

print(images[0])
print(len(y))
print(len(x))

x_data = np.array(x)
y_data = np.array(y)

print(x_data.shape)
print(y_data.shape)

y_data = np_utils.to_categorical(y_data)
print(y_data.shape)
num_classes = y_data.shape[1]
print(num_classes)

x_data , y_data = shuffle(x_data,y_data, random_state = 0)

split = int(0.6*(x_data.shape[0]))

x_train = x_data[:split]
x_test = x_data[split:]
y_train = y_data[:split]
y_test = y_data[split:]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()

model.add(Convolution2D(32,3,3,input_shape = (28,28,3)))
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size= (2,2)))
          
model.add(Convolution2D(16,3,3))
model.add(Activation('relu'))

model.add( Flatten() )

model.add( Dropout(0.2) )
model.add(Dense(num_classes))

model.add(Activation('softmax'))

model.summary()

model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
tbcallback=keras.callbacks.TensorBoard(log_dir='./Graph1',histogram_freq=10,write_graph=True,write_images=True)
model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=20,batch_size = 128,callbacks=[tbcallback], shuffle = True )

model.save('Disease_detector_model.h5')