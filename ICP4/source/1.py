from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pandas as pd

#Reading the dataset 'iris'
ds=pd.read_csv("iris.csv")

#Preprocessing
x=ds.drop('class',axis=1)
y=ds['class']

#Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)

#Training the classifier
model = GaussianNB()
model.fit(x_train,y_train)

#Making Predictions
y_pred = model.predict(x_test)

#Evaluating the model
#Compute confusion matrix to evaluate the accuracy of a classification
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))