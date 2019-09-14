# -*- coding: utf-8 -*-
"""
Instructions:
    
 - Import datasets from sklearn and matplotlib.pyplot as plt.
 - Load the digits dataset using the .load_digits() method on datasets.
 - Print the keys and DESCR of digits.
 - Print the shape of images and data keys using the . notation.
 - Display the 1011th image using plt.imshow(). This has been done for you, so hit 'Submit Answer' to see which handwritten digit this happens to be!
"""

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.data.shape)
print(digits.images.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()