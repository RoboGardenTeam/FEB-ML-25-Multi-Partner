import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def manual_knn(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = np.sqrt(np.sum((test_point - train_point)**2))
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        labels = [label for (d, label) in k_nearest]
        prediction = max(set(labels), key=labels.count)
        predictions.append(prediction)
    return predictions

def builtin_knn(X_train, y_train, X_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    k = 3
    y_pred_manual = manual_knn(X_train, y_train, X_test, k)
    print(f"Manual KNN Accuracy: {accuracy_score(y_test, y_pred_manual):.2f}")
    
    y_pred_builtin = builtin_knn(X_train, y_train, X_test, k)
    print(f"Built-in KNN Accuracy: {accuracy_score(y_test, y_pred_builtin):.2f}")