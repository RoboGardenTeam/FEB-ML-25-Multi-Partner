import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
housing = fetch_california_housing()

X_class = iris.data[:, :2]  # Use first two features for visualization
y_class = (iris.target != 0).astype(int)  # Binary classification (Setosa vs. Non-Setosa)

X_reg = housing.data[:, [0]]  # Use only one feature for visualization
y_reg = housing.target

X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

def manual_svm_classification():
    w = np.zeros(X_class_train.shape[1])
    b = 0
    lr = 0.01
    epochs = 1000
    for _ in range(epochs):
        for i in range(len(y_class_train)):
            if y_class_train[i] * (np.dot(w, X_class_train[i]) + b) < 1:
                w += lr * (y_class_train[i] * X_class_train[i] - 0.01 * w)
                b += lr * y_class_train[i]
            else:
                w += lr * (-0.01 * w)
    x_plot = np.linspace(X_class[:, 0].min(), X_class[:, 0].max(), 100)
    y_plot = -(w[0] / w[1]) * x_plot - (b / w[1])
    plt.scatter(X_class_train[:, 0], X_class_train[:, 1], c=y_class_train, cmap='bwr')
    plt.plot(x_plot, y_plot, 'k-', label='Decision Boundary')
    plt.title("Manual SVM Classification")
    plt.legend()
    plt.show()

def manual_svr_regression():
    w = np.zeros(1)
    b = 0
    lr = 0.01
    epsilon = 0.1
    epochs = 1000
    for _ in range(epochs):
        for i in range(len(y_reg_train)):
            error = y_reg_train[i] - (np.dot(w, X_reg_train[i]) + b)
            if abs(error) > epsilon:
                w += lr * X_reg_train[i] * np.sign(error)
                b += lr * np.sign(error)
    y_pred = np.dot(X_reg_test, w) + b
    plt.scatter(X_reg_test, y_reg_test, color='red', label='Actual Data')
    plt.plot(X_reg_test, y_pred, color='blue', label='Manual SVR Prediction')
    plt.fill_between(X_reg_test.ravel(), y_pred - epsilon, y_pred + epsilon, color='gray', alpha=0.3, label="ε-tube")
    plt.title("Manual SVR Regression")
    plt.legend()
    plt.show()

def sklearn_svm_classification():
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_class_train, y_class_train)
    w = model.coef_[0]
    b = model.intercept_[0]
    x_plot = np.linspace(X_class[:, 0].min(), X_class[:, 0].max(), 100)
    y_plot = -(w[0] / w[1]) * x_plot - (b / w[1])
    plt.scatter(X_class_train[:, 0], X_class_train[:, 1], c=y_class_train, cmap='bwr')
    plt.plot(x_plot, y_plot, 'k-', label='Decision Boundary')
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='black', label='Support Vectors')
    plt.title("Sklearn SVM Classification")
    plt.legend()
    plt.show()

def sklearn_svr_regression():
    scaler = StandardScaler()
    X_reg_train_scaled = scaler.fit_transform(X_reg_train)
    X_reg_test_scaled = scaler.transform(X_reg_test)
    model = SVR(kernel='rbf', C=100, epsilon=0.1)
    model.fit(X_reg_train_scaled, y_reg_train)
    y_pred = model.predict(X_reg_test_scaled)
    plt.scatter(X_reg_test, y_reg_test, color='red', label='Actual Data')
    plt.plot(X_reg_test, y_pred, color='blue', label='SVR Prediction')
    plt.title("Sklearn SVR Regression")
    plt.legend()
    plt.show()

manual_svm_classification()
manual_svr_regression()
sklearn_svm_classification()
sklearn_svr_regression()