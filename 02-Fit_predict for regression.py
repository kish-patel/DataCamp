# -*- coding: utf-8 -*-
"""
Instructions:
 - Import LinearRegression from sklearn.linear_model.
 - Create a LinearRegression regressor called reg.
 - Set up the prediction space to range from the minimum to the maximum of X_fertility. This has been done for you.
 - Fit the regressor to the data (X_fertility and y) and compute its predictions using the .predict() method and the prediction_space array.
 - Compute and print the R2 score using the .score() method.
 - Overlay the plot with your linear regression line. This has been done for you, so hit 'Submit Answer' to see the result!
"""

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()