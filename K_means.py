"""

## Part 1 - K-Means (50%)

implementation of K-Means algorithm - based on the numpy package.**

### **1. Useful packages**
"""

import matplotlib.pyplot as plt
import numpy as np

"""### **2. Data**

Generate a 2D dataset
"""

## Generate 2D data
X = np.concatenate([
    np.random.normal([0, 0], size=(500, 2)),
    np.random.normal([5, 5], size=(500, 2)),
    np.random.normal([5, 0], size=(500, 2)),
    np.random.normal([0, 5], size=(500, 2)),
])

# Shuffle the data
np.random.shuffle(X)

"""Plot the data to explore how many clusters it contains"""

plt.scatter(X[:, 0], X[:, 1], cmap='viridis')

"""### **3. K-Means**

Implement the algorithm
"""

class KMeans():
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        ######## Helper fields #########
        self.X_fit_ = None      # store the dataset X.
                                # useful for the next tasks.

        self.labels_ = None     # store the final labels.
                                # That is, the clusters indices for all the samples

        self.centroids = None   # Here you should store the final centroids.

        self.labels_history = []    # Here you should store the labels of each iteration.
                                    # This can also be useful later.

        self.centroids_history = [] # Here you should store the centroids of each iteration.
                                    # This can also be useful later.

        self.costs = []             # Here you should store the costs of the iterations.
                                    # That is, you should calculate the cost in every iteration
                                    # and store it in this list.

    def fit(self, X):
        self.X_fit_ = X
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        for i in range(self.max_iter):
          distances = self._get_distances(X)
          labels = self._get_labels(distances)
          self.labels_history.append(labels)
          self.centroids_history.append(self.centroids)
          cost = self._calculate_cost(distances)
          self.costs.append(cost)
          if i > 0 and np.array_equal(labels, self.labels_history[i-1]):
            break
          self.centroids = self._get_centroids(X, labels)
        self.labels_ = labels
        self.centroids_history.append(self.centroids)
        self.costs.append(self._calculate_cost(self._get_distances(X)))


    def predict(self, X):
        distances = self._get_distances(X)
        labels = self._get_labels(distances)
        return labels

    def _get_distances(self, X):
       distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
       return distances


    def _get_labels(self, X):
        labels = X.argmin(axis=1)
        return labels

    def _get_centroids(self, X, labels):
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
        return new_centroids

    def _calculate_cost(self, X):
        cost = (X.min(axis=1)).sum()
        return cost

"""Run the algorithm on the 2D dataset"""

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

"""graph plotting of the costs as a function of the iterations"""

costs = kmeans.costs
plt.plot(costs)
plt.show()

"""
plot the clusters and the locations of the centroids at each iteration
"""
fig, axs = plt.subplots(1, len(kmeans.labels_history), figsize=(15, 5))
for i, (ax, labels, centroids) in enumerate(zip(axs, kmeans.labels_history, kmeans.centroids_history)):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k')
    ax.set_title(f"Iteration {i}")
plt.show()
