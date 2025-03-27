from processing import ProcessData
import matplotlib.pyplot as plt
import numpy as np


class KMeansClustering:
    def __init__(self, k=3, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
    def fit(self, X):
        np.random.seed(42)
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for itr in range(self.max_iter):
            cluster_points = np.array([np.argmin(np.linalg.norm(x - centroids, axis=1)) for x in X])
            new_centroids = np.array([X[cluster_points == i].mean(axis=0) for i in range(self.k)])
            if np.all(np.linalg.norm(centroids - new_centroids, axis=1) < self.tol):
                break
            centroids = new_centroids
        return cluster_points, centroids
    def plot(self, X, clusters, iteration):
        plt.figure(figsize=(6, 5))
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolors='k', alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Centroids')
        plt.title(f"Iteration {iteration+1}")
        plt.legend()
        plt.show()
if __name__ == "__main__":
    X = ProcessData()
    kmeans = KMeansClustering()
    clusters, centroids = kmeans.fit(X)