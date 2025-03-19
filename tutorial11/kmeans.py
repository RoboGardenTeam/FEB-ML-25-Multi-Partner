import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=42)

def kmeans(X, k, max_iters=100, tol=1e-4):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Initial Centroids')
    plt.title("Initial Data Distribution (Moderately Scattered)")
    plt.legend()
    plt.show()
    
    for iteration in range(max_iters):
        # Step 2: Assign each point to the nearest centroid
        clusters = np.array([np.argmin(np.linalg.norm(x - centroids, axis=1)) for x in X])
        
        # Step 3: Compute new centroids
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
        
        plt.figure(figsize=(6, 5))
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolors='k', alpha=0.6)
        plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Centroids')
        plt.title(f"Iteration {iteration+1}")
        plt.legend()
        plt.show()
        
        # Step 4: Check convergence
        if np.all(np.linalg.norm(centroids - new_centroids, axis=1) < tol):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

k = 3
clusters, centroids = kmeans(X, k)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolors='k', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Final Centroids')
plt.title("Final Clustering Result")
plt.legend()
plt.show()
