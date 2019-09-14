# -*- coding: utf-8 -*-
"""
#### 
The data has been pre-loaded into a DataFrame called df

Instructions:
 - Import KNeighborsClassifier from sklearn.neighbors.
 - Create arrays X and y for the features and the target variable. Here this has been done for you. Note the use of .drop() to drop the target variable 'party' from the feature array X as well as the use of the .values attribute to ensure X and y are NumPy arrays. Without using .values, X and y are a DataFrame and Series respectively; the scikit-learn API will accept them in this form also as long as they are of the right shape.
 - Instantiate a KNeighborsClassifier called knn with 6 neighbors by specifying the n_neighbors parameter.
 - Fit the classifier to the data using the .fit() method.
"""

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)