# Creating a Notebook in Azure Machine Learning
## Used the following documentation https://learn.microsoft.com/en-us/training/modules/analyze-climate-data-with-azure-notebooks/
## Created a Scatter plot in Azure Notebooks using the Python - Azure ML module.

Code: import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()
yearsBase, meanBase = np.loadtxt('graph (1).csv', delimiter=',', usecols=(0, 1), unpack=True, skiprows=1)
years, mean = np.loadtxt('graph (2).csv', delimiter=',', usecols=(0, 1), unpack=True, skiprows=1)
plt.scatter(yearsBase, meanBase)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
plt.show()


![image](https://github.com/user-attachments/assets/16d1ab47-e674-4f3e-9ac8-66a7bb6337a2)

# Next was a scatter plot with linear regression

## Creates a linear regression from the data points
m,b = np.polyfit(yearsBase, meanBase, 1)

## This is a simple y = mx + b line function
def f(x):
    return m*x + b

## This generates the same scatter plot as before, but adds a line plot using the function above
plt.scatter(yearsBase, meanBase)
plt.plot(yearsBase, f(yearsBase))
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
plt.show()

## Prints text to the screen showing the computed values of m and b
print(' y = {0} * x + {1}'.format(m, b))
plt.show()

![image](https://github.com/user-attachments/assets/176a988f-d360-442a-9d5d-1291e56f4615)


# Linear Regression with Scikit-Learn

# Pick the Linear Regression model and instantiate it
model = LinearRegression(fit_intercept=True)

# Fit/build the model
model.fit(yearsBase[:, np.newaxis], meanBase)
mean_predicted = model.predict(yearsBase[:, np.newaxis])

# Generate a plot like the one in the previous exercise
plt.scatter(yearsBase, meanBase)
plt.plot(yearsBase, mean_predicted)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
plt.show()

print(' y = {0} * x + {1}'.format(model.coef_[0], model.intercept_))

![image](https://github.com/user-attachments/assets/dbb1e142-29fd-459c-bfa3-79523ae7adb3)


