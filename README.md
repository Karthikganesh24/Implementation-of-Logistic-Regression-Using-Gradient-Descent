# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Scikit-learn, and the Iris dataset.
2. Load the Iris dataset and split it into training and testing datasets.
3. Create and train the SGD Classifier model using the training dataset.
4. Predict the Iris flower species using the test data and display the predicted results.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Create dataset
X = np.array([[1], [2], [3], [4], [5],
              [6], [7], [8], [9], [10]])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.1
iterations = 1000
m = len(y)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(iterations):

    z = w * X.flatten() + b
    y_pred = sigmoid(z)

    dw = np.sum((y_pred - y) * X.flatten()) / m
    db = np.sum(y_pred - y) / m

    w = w - learning_rate * dw
    b = b - learning_rate * db

# Final predictions
probabilities = sigmoid(w * X.flatten() + b)
predictions = (probabilities >= 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y, predictions)

print("Logistic Regression Using Gradient Descent")
print("------------------------------------------")
print("Weight:", round(w, 4))
print("Bias:", round(b, 4))
print("Actual Output:", y)
print("Predicted Output:", predictions)
print("Accuracy:", accuracy * 100, "%")

# Plot results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, probabilities, color='red',
         label='Predicted Probability')
plt.xlabel("Input")
plt.ylabel("Probability")
plt.title("Logistic Regression Using Gradient Descent")
plt.legend()
plt.show()

```

## Output:
![alt text](<output/Screenshot 2026-11-25 100200.png>)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.