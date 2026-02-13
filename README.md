# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset.
2. Remove unnecessary columns and convert categorical data using dummy variables.
3. Separate the dataset into features (X) and target variable (Y).
4. Split the data into training and testing sets.
5. Create and train the Linear Regression model using training data.
6. Evaluate the model using 5-fold cross-validation and calculate average R² score.
7. Predict test data values and compute MSE, MAE, and R²; plot actual vs predicted prices.
   

## Program:
~~~
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: RAGHUL.S
RegisterNumber: 212225040325  
*/
/*
Program to implement SGD Regressor for linear regression.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

#load the data set
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

#data preprocessing,dropping the unnecessary coloumn and handling the catergorical variables
data=data.drop(['CarName','car_ID'],axis=1)
data = pd.get_dummies(data, drop_first=True)

#splitting the data 
X=data.drop('price', axis=1)
Y=data['price']

scaler = StandardScaler()
X=scaler.fit_transform(X)
Y=scaler.fit_transform(np.array(Y).reshape(-1, 1))

#splitting the dataset into training and tests
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#create sdg regressor model
sgd_model= SGDRegressor(max_iter=1000, tol=1e-3)

#fiting the model to training data
sgd_model.fit(X_train, Y_train)

#making predictions
y_pred = sgd_model.predict(X_test)

#evaluating model performance
mse = mean_squared_error(Y_test, y_pred)
r2=r2_score(Y_test,y_pred)
mae= mean_absolute_error(Y_test, y_pred)

#print evaluation metrics
print('Name:RAGHUL.S')
print('Reg no: 212225040325')
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-Squared Score:",r2)

#print model coefficients
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

#visualising actual vs predicted prices
plt.scatter(Y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(Y_test),max(Y_test)],[min(Y_test),max(Y_test)],color='red')
plt.show()
~~~

## Output:
![WhatsApp Image 2026-02-12 at 12 11 42 AM](https://github.com/user-attachments/assets/d0fbef63-b879-473a-acb6-fca70d78bd4b)

![EX3 NAME](https://github.com/user-attachments/assets/a1aa2991-8ffc-403c-bcbe-9ac6cd89cba6)

![EX3 PERFORMANCE](https://github.com/user-attachments/assets/b6cfbba0-9e65-40c4-9bae-dcf8344e5087)

![EX3 GRAPH](https://github.com/user-attachments/assets/837c79a7-830e-4f2e-b091-3f4613329ee3)



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
