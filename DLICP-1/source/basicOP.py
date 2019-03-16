import pandas
import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("Breas Cancer.csv")
#print(type(dataset))

x1 = dataset.drop(['id','diagnosis','fractal_dimension_worst'],axis=1)
y1 = dataset['diagnosis']
df = pd.DataFrame(y1)
y1 = df.replace(['M','B'],[0,1])
print(y1)
#y1 = pd.get_dummies(y1)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(x1, y1,
                                                    test_size=0.25, random_state=25)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(100, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(6, activation='sigmoid'))
my_first_nn.add(Dense(6, activation='sigmoid'))
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))
