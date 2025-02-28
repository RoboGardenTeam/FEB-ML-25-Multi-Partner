import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def manual_linear_regression(X_train, y_train, X_test, lr=0.01, iterations=1000):
    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    theta = np.zeros(X_train.shape[1])
    
    for _ in range(iterations):
        pred = X_train.dot(theta)
        error = pred - y_train
        gradient = X_train.T.dot(error)/len(y_train)
        theta -= lr * gradient
    
    return np.where(X_test.dot(theta) >= 0.5, 1, 0)

def builtin_linear_regression(X_train, y_train, X_test):
    model = LinearRegression().fit(X_train, y_train)
    return np.where(model.predict(X_test) >= 0.5, 1, 0)

def manual_logistic_regression(X_train, y_train, X_test, lr=0.01, iterations=5000):
    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    theta = np.zeros(X_train.shape[1])
    
    for _ in range(iterations):
        z = X_train.dot(theta)
        h = 1/(1 + np.exp(-z))
        gradient = X_train.T.dot(h - y_train)/len(y_train)
        theta -= lr * gradient
    
    return np.where(1/(1 + np.exp(-X_test.dot(theta))) >= 0.5, 1, 0)

def builtin_logistic_regression(X_train, y_train, X_test):
    return LogisticRegression(max_iter=10000).fit(X_train, y_train).predict(X_test)

if __name__ == "__main__":
    data = load_digits()
    X = data.data
    y = data.target
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Manual Linear Accuracy:", 
          accuracy_score(y_test, manual_linear_regression(X_train, y_train, X_test)))
    print("Built-in Linear Accuracy:", 
          accuracy_score(y_test, builtin_linear_regression(X_train, y_train, X_test)))
    print("Manual Logistic Accuracy:", 
          accuracy_score(y_test, manual_logistic_regression(X_train, y_train, X_test)))
    print("Built-in Logistic Accuracy:", 
          accuracy_score(y_test, builtin_logistic_regression(X_train, y_train, X_test)))