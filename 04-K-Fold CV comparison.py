# -*- coding: utf-8 -*-
"""
Instructions:
 - Import LinearRegression from sklearn.linear_model and cross_val_score from sklearn.model_selection.
 - Create a linear regression regressor called reg.
 - Use the cross_val_score() function to perform 5-fold cross-validation on X and y.
 - Compute and print the average cross-validation score. You can use NumPy's mean() function to compute the average.
"""

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

"""
Instructions:
    
 - Import LinearRegression from sklearn.linear_model and cross_val_score from sklearn.model_selection.
 - Create a linear regression regressor called reg.
 - Perform 3-fold CV and then 10-fold CV. Compare the resulting mean score
"""
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg,X,y,cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg,X,y,cv=10)
print(np.mean(cvscores_10))