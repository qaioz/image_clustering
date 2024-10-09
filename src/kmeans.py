import numpy as np
from typing import Literal


def log(func, message=None):
    print(f"From {func.__name__}: {message}")


def kmeans(
    image: np.ndarray,
    *,
    num_clusters: int,
    max_iterations: int,
    norm: float | None = None,
    threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    kmeans clustering algorithm

    :param image: np.ndarray of shape (m, n, 3), the input image

    :param num_clusters: int, the number of clusters

    :param max_iterations: int, the maximum number of iterations

    :param norm: works as the argument 'ord' in numpy.linalg.norm function

    :param threshold: float, cost threshold. if new_cost >= current_cost - threshold, then the algorithm stops. I.e.
    If the cost of new clusters is less than the cost of the current clusters minus the threshold, the new clusters will be used, otherwise the algorithm iteration will stop.

    :return: tuple of np.ndarray, np.ndarray (centroids, point_clusters) where
    - centroids: is an array of shape (k, 3) containing the centroids of the clusters
    - point_clusters: is an array of shape (m, n) containing the index of the centroid each pixel is assigned to
    
    I.E if kmeans(image, k=2, max_iterations=10, norm=2, threshold=1) returns (centroids, point_clusters)
    and point_clusters[i, j] = x, then the pixel at image[i, j] is assigned to the centroid centroids[x]

    """

    log(kmeans, "Starting kmeans")

    current_iteration = 1
    centroids = select_centroids(image, num_clusters)
    log(kmeans, f"Selected centroids: {centroids}")
    elements_per_centroid, point_clusters = partition(image, centroids, norm)
    log(kmeans, "Initial partition done")
    current_cost = cost_function(centroids, elements_per_centroid, norm)
    log(kmeans, f"Initial cost: {current_cost}")

    while current_iteration < max_iterations:

        new_centroids = generate_new_centroids(elements_per_centroid)
        new_centroid_map, new_point_clusters = partition(image, new_centroids, norm)
        new_cost = cost_function(new_centroids, new_centroid_map, norm)

        if new_cost < current_cost - threshold:
            centroids = new_centroids
            elements_per_centroid = new_centroid_map
            point_clusters = new_point_clusters
            current_cost = new_cost
        else:
            if new_cost >= current_cost:
                log(kmeans, f"New cost: {new_cost} is greater than current cost: {current_cost}")
            elif new_cost >= current_cost - threshold:
                log(kmeans, f"New cost: {new_cost} is within threshold of current cost: {current_cost}")
            break

        current_iteration += 1

        log(kmeans, f"Current iteration: {current_iteration}")
        log(kmeans, f"Current cost: {current_cost}")
        # log(kmeans, f"Current centroids: {centroids}")
    log(kmeans, "Kmeans done")
    log(kmeans, f"final clusters: {centroids}")
    return centroids, point_clusters



def generate_new_centroids(cluster_points: list[np.ndarray]) -> np.ndarray:
    """
    Chooses new centroids based on the average distance of the points in the cluster to its current centroid

    :param cluster_points: len(cluster_points) == k ==> k is the number of clusters.
    [r,g,b] \in cluster_points[i] 
    
    
    """
    new_centroids = [None] * len(cluster_points)

    for i, cluster in enumerate(cluster_points):
        new_centroids[i] = np.mean(cluster, axis=0)

    return np.array(new_centroids)


def partition(
    image: np.ndarray, centroids: np.ndarray, norm: float | None
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Assign each pixel in the image to the nearest centroid and return both the pixel clusters and
    a map of the image coordinates to the corresponding clusters.

    param: image: np.ndarray of shape (m, n, 3), the input image

    param: centroids: np.ndarray of shape (k, 3), where k is the number of centroids

    param: norm: the norm to use to calculate the distance between the pixel and the centroid

    returns:
        - list of np.ndarray, where each array contains the pixels that belong to a specific cluster.
        - np.ndarray of shape (m, n), a coordinate map where each entry points to the index of the centroid the pixel is assigned to.
    """

    print("Partitioning the image")

    m, n, _ = image.shape
    k = centroids.shape[0]

    # Flatten the image to a 2D array of shape (m * n, 3)
    flat_image = image.reshape(-1, 3)

    # Compute the distances between each pixel and each centroid
    differences = flat_image[:, np.newaxis] - centroids

    # Compute the norms of the differences
    norms = np.linalg.norm(differences, norm, axis=2)

    # Find the index of the centroid with the minimum norm for each pixel
    min_norms = np.argmin(
        norms, axis=1
    )  # Array of shape (m * n) containing centroid indices

    # Create a map of the centroids to the pixels
    centroids_map = [[] for _ in range(k)]
    for i in range(m * n):
        centroids_map[min_norms[i]].append(flat_image[i])

    # Create clusters: a list of np.ndarrays, one for each cluster
    clusters = [np.array(cluster) for cluster in centroids_map]

    # Reshape the min_norms array to match the original image shape (m, n)
    coordinate_clusters = min_norms.reshape(m, n)

    # Return the clusters and the coordinate_clusters map
    return clusters, coordinate_clusters


def select_centroids(image: np.ndarray, k: int) -> np.ndarray:
    """
    Select k random centroids from the image.

    each centroid is a color rgb which is of a  shape (3,), so the return value is a np.ndarray of shape (k, 3)

    This function is very cheap, so I will not use numpy vectorization
    """

    m, n, _ = image.shape
    centroids = []

    for _ in range(k):
        x = np.random.randint(0, m)
        y = np.random.randint(0, n)
        while tuple(image[x, y]) in centroids:
            x = np.random.randint(0, m)
            y = np.random.randint(0, n)

        centroids.append(tuple(image[x, y]))

    return np.array(centroids)


def cost_function(
    centroids: np.ndarray, elements_per_centroid: list[np.ndarray], norm: float | None
) -> float:
    """
    Calculate the cost function for the kmeans algorithm

    param: centroids: np.ndarray of shape (k, 3)

    param: points_per_centroid: list of ndarrays of shape (x, 3) where
    length of the list is the number of centroids and x is the number of elements in the cluster. Note that each element
    already represents a color

    returns: float
    """

    # calculate the difference between the centroid and the points in the cluster using vectorization
    # print difference each dimension with np ndarray functions
    cost = 0
    for i in range(len(elements_per_centroid)):
        norm_diff = np.linalg.norm(
            elements_per_centroid[i] - centroids[i][np.newaxis], norm, axis=1
        )
        cost += np.sum(norm_diff)

    return cost
