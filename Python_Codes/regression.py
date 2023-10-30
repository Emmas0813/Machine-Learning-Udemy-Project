#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------

# importing dataset

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values 

# -----------------------

# splitting dataset into training and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1) #20% in test set and 80% in training set

# training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting test results
y_pred = regressor.predict(x_test)

# visualizing the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs. Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#prediciting test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs. Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()