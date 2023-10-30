

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values # note, ':' is a range and ': - 1' is everything but the last column (aka dependent variable)
y = dataset.iloc[:, -1].values # values of only the last column
# ----------------------
print(x)
# ----------------------
print(y)

#taking care of missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = mean) # replaces missing data with the average
imputer.fit(X[:, 1:3]) # not inclusive so need to put upper bound as higher column.
X[:, 1:3] = imputer.transform(X[:, 1:3]) # replace the nan values

# ---------------------
print(x)

# encoding the independent variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder(), [0])], remainder='passthrough') # encode the column of countries with the onehotloader class
x = np.array[ct.fit_transform(x)] # replace the country literals with their encoded data (in this case an array of 3 values)

# ---------------------
print(x)

# encoding the dependent variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
y = le.fit_transform(y) # replace y column with encoded data

# ---------------------

print(y)

# ---------------------

# splitting the data set into the training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1) #20% in test set and 80% in training set

# --------------------
print(x_train)
print(x_test)
print(y_train)
print(y_test)
# -------------------

# feature scaling (should come after splitting data)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # performs standarization
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:]) # takes info from column 3 through rest of range
x_test[:, 3:] = sc.transform(x_test[:, 3:])
