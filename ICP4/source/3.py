from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
#Reading 'iris' dataset
dataset=pd.read_csv("iris.csv")
#Preprocessing
x=dataset.drop('class',axis=1)
y=dataset['class']
#Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
#Training the Gaussian Naive bayes classifier
model=GaussianNB()
model.fit(x_train,y_train)
#Making Predictions
y_pred1=model.predict(x_test)
classifier=SVC(kernel='linear',C=1).fit(x_train,y_train)
#Making predictions
y_pred2=classifier.predict(x_test)
#Evaluating results for both the classifiers.
print(classification_report(y_test,y_pred1))
print("Gaussian Naive Bayes model accuracy is: ",metrics.accuracy_score(y_test,y_pred1)*100,"%")

print(classification_report(y_test,y_pred2))
print("Linear SVM classifier accuracy is: ",metrics.accuracy_score(y_test,y_pred2)*100,"%")