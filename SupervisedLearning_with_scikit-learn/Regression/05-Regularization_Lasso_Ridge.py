# -*- coding: utf-8 -*-
"""
Regularization I: Lasso

Instructions:
 - Import Lasso from sklearn.linear_model.
 - Instantiate a Lasso regressor with an alpha of 0.4 and specify normalize=True.
 - Fit the regressor to the data and compute the coefficients using the coef_ attribute.
 - Plot the coefficients on the y-axis and column names on the x-axis. This has been done for you, so hit 'Submit Answer' to view the plot!
"""
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


"""
Regularization II: Ridge

Instructions:  
 - Instantiate a Ridge regressor and specify normalize=True.
 - Inside the for loop:
     - Specify the alpha value for the regressor to use.
     - Perform 10-fold cross-validation on the regressor with the specified alpha. The data is available in the arrays X and y.
     - Append the average and the standard deviation of the computed cross-validated scores. NumPy has been pre-imported for you as np.
 - Use the display_plot() function to visualize the scores and standard deviations.
 """
 
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
