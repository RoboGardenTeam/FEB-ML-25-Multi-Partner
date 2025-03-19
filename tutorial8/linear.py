import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the Iris dataset (common built-in dataset)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features for simplicity
print(X[:10, :])
y = (iris.target == 0).astype(int)  # Binary classification: Class 0 vs Not Class 0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Manual Linear Regression for Classification
class ManualLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []
        #y = mX + b

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y
            self.errors.append(np.mean(np.square(error)))  # Store MSE for each epoch

            # Gradient Descent
            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return (y_predicted >= 0.5).astype(int)

# Train and Evaluate Manual Model
manual_model = ManualLinearRegression()
manual_model.fit(X_train, y_train)
manual_predictions = manual_model.predict(X_test)
manual_accuracy = accuracy_score(y_test, manual_predictions) * 100
print("Manual Linear Regression Accuracy: {:.2f}%".format(manual_accuracy))

# Print Inputs and Predictions
print("\nInputs, Actual Classifications, and Predictions:")
for i in range(len(X_test)):
    print(f"Input: {X_test[i]}, Actual: {y_test[i]}, Predicted: {manual_predictions[i]}")

# Built-in Linear Regression
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_predictions = (sklearn_model.predict(X_test) >= 0.5).astype(int)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions) * 100
print("\nBuilt-in Linear Regression Accuracy: {:.2f}%".format(sklearn_accuracy))

# Plot Decision Boundaries
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='Actual')
plt.scatter(X_test[:, 0], X_test[:, 1], c=manual_predictions, cmap='coolwarm', marker='x', label='Predicted')
plt.title("Manual Linear Regression Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Plot Error vs Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(len(manual_model.errors)), manual_model.errors, color='red')
plt.title("Training Error vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()  

# Confusion Matrix
cm = confusion_matrix(y_test, manual_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Class 0", "Class 0"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Manual Linear Regression")
plt.show()