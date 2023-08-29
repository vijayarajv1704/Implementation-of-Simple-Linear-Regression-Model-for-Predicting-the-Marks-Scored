# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vijayaraj V
RegisterNumber:  212222230174
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


```

## Output:

![image](https://github.com/vijayarajv1704/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121303741/ba10bcc8-e402-43fb-8651-953ae9cffb77)

![image](https://github.com/vijayarajv1704/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121303741/c7e671bd-083c-4c5a-b677-800524768b86)

![image](https://github.com/vijayarajv1704/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121303741/a5d05272-537b-407a-b4bc-a974e7790639)

![image](https://github.com/vijayarajv1704/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121303741/109b3fbc-6e8d-4cf2-9861-336f54aaeb73)

![image](https://github.com/vijayarajv1704/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121303741/b65216d2-facf-4eea-856c-7997bf970f65)

![image](https://github.com/vijayarajv1704/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121303741/4db02849-530f-4697-ac29-06180776c42c)

![image](https://github.com/vijayarajv1704/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121303741/976f779b-ed74-435a-acbb-6be83b043dfd)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
