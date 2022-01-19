import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def initializecenters(x, k):
    centers = []
    for i in range(k):
        index = random.randint(0, len(x))
        centers.append(x[index])
    return centers

# calculate euclidean distance
def calculateDistance(x, y):
    distance = 0
    for i in range(len(x)):
        distance += (x[i] - y[i])**2
    return distance**0.5

def findClosestCenters(x, centers):
    x_index = []
    for point in x:
        distances = []
        for center in centers:
            distances.append(calculateDistance(point, center))
        x_index.append(np.argmin(distances))
    return x_index

def computeMeans(x, x_index, k, size, dimension):
    centers = []
    for i in range(k):
        indices = [j for j, cluster in enumerate(x_index) if cluster == i] # get the indices of points in dataset belong to the same cluster
        sum_dimensions = [0] * dimension  # maintain sum of values of each dimension
        for idx in indices:
            for d in range(dimension):
                sum_dimensions[d] += x[idx][d]
        mean = []
        for el in sum_dimensions:
            mean.append(el / len(indices))
        centers.append(mean)
    return centers

if __name__ == "__main__":
    dataset = pd.read_csv("Dataset1.csv")
    size, dimension = dataset.shape
    x = np.array(dataset.values)

    # plot the whole data set
    x_dimension = []
    y_dimension = []
    for i in range(size):
        x_dimension.append(x[i][0])
        y_dimension.append(x[i][1])
    plt.scatter(x_dimension, y_dimension)
    plt.title("Dataset1")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    k = 2 # number of clusters
    centers = initializecenters(x, k)

    n = 15 # number of iterations
    for i in range(n):
        x_index = findClosestCenters(x, centers)
        centers = computeMeans(x, x_index, k, size, dimension)

    # plot the result of clustering
    for i in range(k):
        indices = [j for j, cluster in enumerate(x_index) if cluster == i] # get the indices of points in dataset belong to the same cluster
        x_dimension = []
        y_dimension = []
        for idx in indices:
            x_dimension.append(x[idx][0])
            y_dimension.append(x[idx][1])
        plt.scatter(x_dimension, y_dimension)
        plt.scatter(centers[i][0], centers[i][1], marker="^", s=200)
    plt.title("K-Means Clusteringو K=2")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
