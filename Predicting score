#Predict the percentage of an student based on the no. of study hours.

# Importing all libraries required 
import pandas as pd # for reading the data set and converting it to dataframe
import numpy as np  
import matplotlib.pyplot as plt  # for plotting the different graphs
%matplotlib inline

student_data = pd.read_csv("http://bit.ly/w-data") # loading the data from the link given by the TSF

student_data.shape # checking the records present in the dataset which gives rows and columns of the dataset

student_data.head() # will give first five records present in the dataset

student_data.describe() # will give the different statistical values of the dataset

student_data.corr() # correlation of the data set i.e relation between the two variables

student_data.info() # gives the details of the dataset

student_data.plot(kind='scatter',x='Hours',y='Scores')
plt.title("Hours vs Scores")    # plotting the scatter plot for the dataset

X = student_data.iloc[:, :-1].values  
Y = student_data.iloc[:, 1].values   # storing  the variables valuesin the another variable 

#Splitting the data set as Training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, random_state=0)


#Fitting  simple linear Regression using Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)

#Predicting the values of test set
Score_prediction = model.predict(X_train)

print(model.coef_) # printing the co-efficient

print(model.intercept_) # intercept of the model

#Visualizing the Training data set results using graph
plt.scatter(X_train, Y_train, color= 'blue')
plt.plot(X_train, model.predict(X_train), color = 'red')
plt.title('Hours vs Scores(Training Data Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

#Predicting the value of score using hour that is alredy in the data set
model.predict([[5.1]])

#Actual values of the  test set
Y_test

#Calculating the predicted values of the X_test
Y_predict=model.predict(X_test)

Y_predict

#Data frame for the Actual and predicted values
dfrm= pd.DataFrame(Y_test,Y_predict)

df1 = pd.DataFrame({'Actual':Y_test, 'Predicted':Y_predict})
df1

#Given Test data
Hrs = 9.25
predct = model.predict([[Hrs]])
print("Number of hours: ",Hrs)
print("Predicted scores: ",predct)

#Random Test data
Hrs1 = 8
predct1 = model.predict([[Hrs1]])
print("Number of hours: ",Hrs1)
print("Predicted scores: ",predct1)

import sklearn.metrics as metrics 
from sklearn.metrics import confusion_matrix, accuracy_score
import math
print("Mean Absolute error :",metrics.mean_absolute_error(Y_test,Y_predict))#It will give the mean errorof  the model
print("Mean Squared Error:",metrics.mean_squared_error(Y_test,Y_predict))
print("Root Mean Squared Error:",math.sqrt(metrics.mean_squared_error(Y_test,Y_predict)))

model.score(X_test,Y_test) # checking the score of the model
