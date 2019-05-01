from __future__ import print_function
import numpy as np
import itertools
import pandas as pd
import keras_metrics
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import metrics
from sklearn.metrics import confusion_matrix
from keras import regularizers
from sklearn.metrics import roc_curve, auc, precision_score, accuracy_score, recall_score

import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = "Times New Roman"

np.random.seed(306)  # for reproducibility

# network and training
NB_EPOCH = 90
BATCH_SIZE = 512
VERBOSE = 0
NB_CLASSES = 2   # Heart Disease diagnosis yes = 1, no = 0
OPTIMIZER = RMSprop(lr=0.001) # optimizer
N_HIDDEN = 128
TRAINING_SPLIT = 0.6 # how much from all of the data is split for training
VALIDATION_SPLIT=0.4 # how much in TRAIN is reserved for VALIDATION
DROPOUT = 0.5

#DATA PREPARATION
# read synthetic cleveland dataset from full cleveland.data
df_main = pd.read_csv("heart.csv", sep=',')

# Neural Net transfer function likes to work with floats
df_main.astype(float)

# Normalize values to range [0:1]
df_main /= df_main.max()

# split data into independent and dependent variables
y_all = df_main['target']
X_all = df_main.drop(columns = 'target')

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size = TRAINING_SPLIT, test_size = 1 - TRAINING_SPLIT)
Y_train = np_utils.to_categorical( y_train, NB_CLASSES)
Y_test = np_utils.to_categorical( y_test, NB_CLASSES)



# set up the NN model, Dropout & L2 regularization
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(13,), kernel_regularizer = regularizers.l2( 0.01)))
model.add(Activation('tanh'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('tanh'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])

print(model.metrics_names) # metrics to retrieve from score after evaluation

# train the model
from keras.callbacks import TensorBoard
tbc = TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=BATCH_SIZE,
                         write_images=True)
history = model.fit(X_train, Y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH,verbose=VERBOSE, validation_split=VALIDATION_SPLIT,callbacks=[tbc])

# make prediction for ROC
y_pred = model.predict_classes(X_test)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('\nFinal test scores:')
print('\nTest Loss:', score[0])
print('Test Accuracy:', score[1])
print('Test Precision:', score[2])
print('Test Recall:', score[3])

# list all data in history
#print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for precision
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model Precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for recall
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model Recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix
class_names = ['heart_disease', 'no_heart_disease']
cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# plot ROC Curve
predictions = model.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test,predictions[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
#plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()


# plot boxplot accuracy, precision, recall
bplot = [history.history['acc'], history.history['precision'], history.history['recall']]
labels = ['Accuracy', 'Precision', 'Recall']

colors = ['orange', 'cyan', 'red']
fig = plt.figure(figsize=(10,10))
fig.suptitle('Keras Deep Learning Model Performance', fontsize=40)

a = fig.add_subplot(111)
a.tick_params(axis = 'both', which = 'major', labelsize = 40)
a.set_ylim(0,1)
bplot = a.boxplot(bplot, labels=labels,
                  vert=True, patch_artist=True)

# color the boxes
for bp in bplot:
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
a.yaxis.grid(True)
a.set_xlabel('Keras Deep Learning Metrics',fontsize=40)
a.set_ylabel('Score',fontsize=40)

acc=np.mean(y_test==y_pred)
print( acc)

model.save('heart_disease_predict.h5')