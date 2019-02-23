from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd

#Reading the 'iris' dataset
dataset=pd.read_csv("iris.csv")

#Preprocessing
x=dataset.drop('class',axis=1)
y=dataset['class']

#Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)

#Training the classifier
classifier=SVC(kernel='linear',C=1).fit(x_train,y_train)

#Making Predictions
y_pred=classifier.predict(x_test)

#Evaluating the model
print(confusion_matrix(y_test,y_pred))