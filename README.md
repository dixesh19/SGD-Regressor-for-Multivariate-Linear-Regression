# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the California housing dataset
Select the first 3 features as input (X)
and use house price and number of occupants as target (Y).

2.Split the dataset
Divide the data into training and testing sets using train_test_split.

3.Scale the data
Use StandardScaler to normalize both features (X) and targets (Y) for better convergence.

4.Initialize the model
Use SGDRegressor and wrap it with MultiOutputRegressor to handle multiple outputs.

5.Train the model
Fit the model on the scaled training data (X_train, Y_train).

6.Make predictions
Predict on the test set (X_test), then inverse transform predictions and actual values to original scale.

7.Evaluate the model
Calculate the Mean Squared Error (MSE) between predicted and actual values. Display predictions and MSE.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: DINESH R
RegisterNumber:  212224240037
*/

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
X=data.data[:,:3]
Y=np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.fit_transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.fit_transform(Y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",Y_pred[:5])


```

## Output:
![Screenshot 2024-09-12 110353](https://github.com/user-attachments/assets/853e42af-0a68-4f9d-835b-ee0c7bccc119)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
