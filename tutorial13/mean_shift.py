import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

# Gaussian Kernel Function
def gaussian_kernel(distance, bandwidth):
    return np.exp(- (distance ** 2) / (2 * bandwidth ** 2))

# Mean Shift Algorithm with Visualization
def mean_shift_visualized(X, bandwidth=1.5, max_iter=10, tol=1e-3):
    points = np.copy(X)  # Start with original points
    plt.figure(figsize=(12, 8))
    for i in range(max_iter):
        new_points = []
        for p in points:
            distances = cdist([p], X)[0]  # Compute distance to all points
            weights = gaussian_kernel(distances, bandwidth)  # Compute Gaussian weights
            new_p = np.sum(X * weights[:, np.newaxis], axis=0) / np.sum(weights)  # Weighted mean
            new_points.append(new_p)

        new_points = np.array(new_points)

        # Plot the movement of points
        plt.subplot(3, 4, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, label="Original Data")
        plt.scatter(points[:, 0], points[:, 1], color='blue', label="Before Shift", s=10)
        plt.scatter(new_points[:, 0], new_points[:, 1], color='red', label="After Shift", s=10)
        
        # Draw arrows showing movement
        for old, new in zip(points, new_points):
            plt.arrow(old[0], old[1], new[0] - old[0], new[1] - old[1], 
                      head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.5)

        plt.title(f"Iteration {i + 1}")
        plt.legend()
        
        # Check for convergence
        shift_distances = np.linalg.norm(new_points - points, axis=1)
        if np.max(shift_distances) < tol:
            break
        points = new_points
    plt.tight_layout()
    plt.show()

    return points

# Step 2: Identify Unique Cluster Centers
def find_clusters(points, bandwidth):
    cluster_centers = []
    for p in points:
        found = False
        for c in cluster_centers:
            if np.linalg.norm(p - c) < bandwidth / 2:  # Merge close points
                found = True
                break
        if not found:
            cluster_centers.append(p)
    return np.array(cluster_centers)

# Step 3: Assign Each Point to a Cluster
def assign_clusters(X, cluster_centers):
    #labels[i] = 1 -> X[i] belongs to cluster 1
    #[1, 2, 3, 4, 6, 7, 8]
    #X[(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
    labels = np.array([np.argmin([np.linalg.norm(p - c) for c in cluster_centers]) for p in X])
    return labels

# Generate dataset with some noise (outliers)
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=1.0, random_state=42)
X_noise = np.random.uniform(low=-10, high=10, size=(30, 2))  # Random noise points
X = np.vstack((X_clusters, X_noise))  # Combine clusters + noise

# Visualize Initial Data
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], color='gray', alpha=0.5, label="Data Points")
plt.title("Initial Dataset with Noise")
plt.legend()
plt.show()

# Run Mean Shift and visualize steps
shifted_points = mean_shift_visualized(X, bandwidth=1.5)

# Identify final cluster centers
cluster_centers = find_clusters(shifted_points, bandwidth=1.5)

# Assign points to the nearest cluster center
labels = assign_clusters(X, cluster_centers)

# Final Clustering Result
plt.figure(figsize=(6, 6))
for i, center in enumerate(cluster_centers):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')
    plt.scatter(center[0], center[1], s=200, c='black', marker='X', edgecolors='white', linewidth=2)
plt.legend()
plt.title("Final Mean Shift Clustering with Outliers")
plt.show()