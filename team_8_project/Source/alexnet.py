import numpy as np
import keras
from keras.preprocessing import image
from keras import optimizers
from keras.layers import MaxPooling2D,Convolution2D,Dropout,Flatten,Dense,Activation
from keras.models import Sequential,save_model
from keras.utils import np_utils
from keras.layers import merge,Input
from keras.models import Model
import os
import cv2
from sklearn.utils import shuffle
from keras.layers.normalization import BatchNormalization
path="/Users/adisekharrajuuppalapati/Downloads/crowdai"
leaf=os.listdir(path)
print(len(leaf),type(leaf))
print(leaf)
print(leaf[0][2:4])
x,y=[],[]
for i in leaf:
    images = os.listdir(path+"/"+i)
    for j in images:
        img_path = path+"/"+i+"/"+j
        #Better method then cv2.imread
        img = image.load_img(img_path, target_size=(224,224))
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

# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Convolution Step 2
classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

# Max Pooling Step 2
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 3
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 4
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 5
classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

# Max Pooling Step 3
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Flattening Step
classifier.add(Flatten())

# Full Connection Step
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 1000, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 38, activation = 'softmax'))
classifier.summary()

classifier.load_weights('best_weights.hdf5')

for i, layer in enumerate(classifier.layers[:20]):
    print(i, layer.name)
    layer.trainable = False
classifier.summary()

classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# checkpoint
from keras.callbacks import ModelCheckpoint
weightpath = "best_weights.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
tbcallback=keras.callbacks.TensorBoard(log_dir='./Graph',histogram_freq=10,write_graph=True,write_images=True)
classifier.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=25,batch_size=128,shuffle=True,callbacks=[tbcallback,checkpoint])
classifier.save('disease_alexnet_detector.h5')
