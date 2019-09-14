# -*- coding: utf-8 -*-
"""
The DataFrame has been pre-loaded as df. 
This time, you will create the feature array X and target variable array y yourself.

Instructions:
 - Create arrays for the features and the target variable from df. As a reminder, the target variable is 'party'.
 - Instantiate a KNeighborsClassifier with 6 neighbors.
 - Fit the classifier to the data.
 - Predict the labels of the training data, X.
 - Predict the label of the new data point X_new.
 
"""

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
