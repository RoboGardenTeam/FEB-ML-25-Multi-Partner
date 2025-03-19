import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def best_split(X, y):
    n_samples, n_features = X.shape
    best_feature, best_threshold = None, None
    best_score = float("inf")  # Lower is better

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue  # Avoid empty splits

            score = gini(y[left_mask]) if is_classification(y) else mse(y[left_mask]) + mse(y[right_mask])
            
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

def is_classification(y):
    return len(np.unique(y)) < 10  # Assume classification if fewer than 10 unique values

def gini(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def mse(y):
    return np.mean((y - np.mean(y)) ** 2)

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0] if is_classification(y) else np.mean(y)

        feature, threshold = best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0] if is_classification(y) else np.mean(y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._grow_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._grow_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0) if not is_classification(predictions[0]) else np.round(np.mean(predictions, axis=0))

if __name__ == "__main__":
    X_class, y_class = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    rf_manual_c = RandomForest(n_trees=10, max_depth=5)
    rf_manual_c.fit(X_train_c, y_train_c)
    y_pred_manual_c = rf_manual_c.predict(X_test_c)
    print("Manual Random Forest Classification Accuracy:", accuracy_score(y_test_c, y_pred_manual_c))

    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    rf_manual_r = RandomForest(n_trees=10, max_depth=5)
    rf_manual_r.fit(X_train_r, y_train_r)
    y_pred_manual_r = rf_manual_r.predict(X_test_r)
    print("Manual Random Forest Regression MSE:", mean_squared_error(y_test_r, y_pred_manual_r))