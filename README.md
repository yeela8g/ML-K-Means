# K-Means - ML algorithm

Implementation of the K-Means clustering algorithm using Python and NumPy.
The K-Means algorithm is a popular unsupervised machine learning technique used for clustering data points into groups based on their similarity. 

## K-Means Algorithm Overview
K-Means is an iterative algorithm that partitions data points into K clusters. The main steps of the algorithm are as follows:

Initialization: Randomly initialize K centroids (Hyperparameter).
Assignment: Assign each data point to the nearest centroid.
Update Centroids: Recalculate the centroids as the mean of the data points assigned to each cluster.
Repeat: Repeat steps 2 and 3 until convergence or a maximum number of iterations is reached.
The algorithm aims to minimize the within-cluster variance, which is the sum of squared distances between each data point and its assigned centroid.

### Useful Packages:
  
  - **Matplotlib**: For data visualization.
  
  - **NumPy**: For numerical computations.

## Data Generation
A 2D dataset is generated for demonstration purposes. It contains four clusters with 500 data points each, generated from normal distributions with different means.

## K-Means Class
The KMeans class is implemented to perform K-Means clustering. It includes methods for fitting the model to the data and making predictions:

  - **fit**: Fits the K-Means model to the provided data. It initializes centroids, assigns data points to clusters, updates centroids iteratively until convergence, and stores necessary information such as labels, centroids, and costs.
  - **predict**: Assigns data points to the nearest centroids based on the trained model.


## Running the Code
To run the provided code:

1. Install the required packages: matplotlib and numpy.
2. Execute the script containing the K-Means implementation.

## Visualizations
The code includes visualizations to demonstrate the clustering process:

Cost vs. Iteration: A graph showing how the cost changes over iterations, helping to monitor convergence.

![image](https://github.com/yeela8g/ML-K-Means/assets/118124478/7535837b-871c-45ca-bcf0-36cfbe5e7bc7)


Cluster Assignments: Plots showing the data points colored by their assigned clusters, along with centroids' positions at each iteration.


![image](https://github.com/yeela8g/ML-K-Means/assets/118124478/13fd9bd3-1040-45ab-958c-44d208e32c73)

