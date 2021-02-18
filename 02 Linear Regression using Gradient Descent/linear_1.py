# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
plt.rcParams['figure.figsize'] = (13.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
m = 0
c = 0

L = 0.001  # The learning Rate
epochs = 10000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print ("w1 = ", m)
print ("w0 = ", c)

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

#Print Predicted value
print()
print("Predicted Value")
print(Y_pred)

#Print mean square error
print()
tempArray = 0
for f in range(len(Y_pred)):
	tempArray = tempArray + np.square(Y[f] - Y_pred[f])
print("Mean Squared Error : ", tempArray/len(Y_pred))

#Print model accuracy
print()
print("R^2 (Accuracy) : ", r2_score(Y, Y_pred))