###########################Density-Based Spatial Clustering of Applications with Noise (DBSCAN)###########################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate synthetic data (moons dataset for non-linearly separable clusters)
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

# DBSCAN parameters
EPSILON = 0.2  # Neighborhood radius
MIN_POINTS = 20  # Minimum points required to form a cluster

# Function to compute neighbors of a point
def get_neighbors(X, index, epsilon):
    neighbors = []
    for i, point in enumerate(X):
        if np.linalg.norm(X[index] - point) <= epsilon:
            neighbors.append(i)
    return neighbors
# DBSCAN Algorithm
def dbscan(X, epsilon, min_pts):
    labels = np.full(len(X), -1)  # Initialize all points as noise (-1)
    cluster_id = 0
    for i in range(len(X)):
        if labels[i] != -1:  # Skip already classified points
            continue
        # Find neighbors
        neighbors = get_neighbors(X, i, epsilon)
        if len(neighbors) < min_pts:  # Mark as noise if not enough neighbors
            continue  # Stay -1 (noise)

        # Start a new cluster
        labels[i] = cluster_id
        queue = set(neighbors)  # Use a set to prevent duplicate processing
        while queue:
            j = queue.pop()

            if labels[j] == -1:  # Noise becomes part of the cluster
                labels[j] = cluster_id

            if labels[j] != -1:  # Skip already processed points
                continue
            #labels[j] = cluster_id  # Assign to the current cluster
            new_neighbors = get_neighbors(X, j, epsilon)

            if len(new_neighbors) >= min_pts:
                queue.update(new_neighbors)  # Expand cluster
        
        cluster_id += 1  # Move to next cluster

    return labels
def run_dbscan(X, eps=0.3, min_samples=3):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
# Run DBSCAN
labels = dbscan(X, EPSILON, MIN_POINTS)

# Plot results
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
for label in unique_labels:
    if label == -1:
        color = 'black'  # Noise points
        marker = 'x'
    else:
        color = colors[label % len(colors)]
        marker = 'o'

    plt.scatter(X[labels == label, 0], X[labels == label, 1], c=color, label=f'Cluster {label}' if label != -1 else "Noise", marker=marker)
plt.title("DBSCAN Clustering (Fixed Implementation)")
plt.legend()
plt.show()