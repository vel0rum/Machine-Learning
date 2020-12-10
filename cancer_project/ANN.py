# Artificial Neural Network

# Importing the libraries
import pytest
import numpy as np
import pandas as pd
from google.colab import drive

import tensorflow as tf

# Part 1 - Data Preprocessing

# Importing the dataset
def import_local_data(file_path):
    """import data file and return pandas dataframe"""
    raw_df = pd.read_csv(file_path)
    return raw_df

local_file_path = 'nan.csv'
raw_data = import_local_data(local_file_path)

"""
Dataset consists of nine features and one binary dependent variable.

age is broken down into six categories.

menopause is broken down into three categories.

tumor-size is broken down into eleven categories. 5-9 and 10-14 were interpreted as dates in the .csv file (05-Sept and Oct-14), so I modified them to what they should be.

inv-nodes is broken down into seven categories. Again, some entries were interpreted as dates and were corrected. There were no categories for people with 18-23 or 27+ inv-nodes.

node-caps is either yes or no, 8 entries had a '?'.

deg-malig is either 1, 2, or 3.

breast is either left or right.

breast-quad is split into five categories with one entry as a '?'.

irradiat is either yes or no.

The dependent variable is either recurrence-events or no-recurrence-events.


"""

# Taking care of missing data
print(raw_data.shape)
raw_data = raw_data.dropna(how='any')
print(raw_data.shape)
print(raw_data.dtypes)

# Splitting into features and independent variable
X = raw_data.iloc[:, :-1].values
y = raw_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Label encoding the "age" column
le_age = LabelEncoder()
X[:, 0] = le_age.fit_transform(X[:, 0])

# Label encoding the "menopause" column
le_menopause = LabelEncoder()
X[:, 1] = le_menopause.fit_transform(X[:, 1])

# Label encoding the "tumor-size" column
le_tumor_size = LabelEncoder()
X[:, 2] = le_tumor_size.fit_transform(X[:, 2])

# Label encoding the "inv-nodes" column
le_inv_nodes = LabelEncoder()
X[:, 3] = le_inv_nodes.fit_transform(X[:, 3])

# Label encoding the "node-caps" column
le_node_caps = LabelEncoder()
X[:, 4] = le_node_caps.fit_transform(X[:, 4])

# Do not need to encode "deg-malig" column

# Label encoding the "breast" column
le_breast = LabelEncoder()
X[:, 6] = le_breast.fit_transform(X[:, 6])

# Label encoding the "irradiant" column
le_irradiant = LabelEncoder()
X[:, 8] = le_irradiant.fit_transform(X[:, 8])

# One Hot Encoding the "breast-quad" column
ct_breast_quad = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7])], remainder='passthrough')
X = np.array(ct_breast_quad.fit_transform(X))

print(X)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
"""
