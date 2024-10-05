import numpy as np
from src.utils import preformance_counter

def log(func, message = None):
    print(f"From {func.__name__}: {message}")


def kmeans(image: np.ndarray, k: int, max_iterations, norm) -> np.ndarray:
    """
    kmeans clustering algorithm

    param: image: np.ndarray of shape (m, n, 3)
    param: k: number of clusters
    param: norm_function: function to calculate the norm between two points

    """

    log(kmeans, "Starting kmeans")
    
    
    current_iteration = 1
    centroids = select_centroids(image, k)
    log(kmeans, f"Selected centroids: {centroids}")
    elements_per_centroid = partition(image, centroids, norm)
    log(kmeans, "Initial partition done")
    current_cost = cost_function(centroids, elements_per_centroid, norm)
    log(kmeans, f"Initial cost: {current_cost}")

    while current_iteration < max_iterations:
        print("Current iteration: ", current_iteration)

        new_centroids = generate_new_centroids(elements_per_centroid)
        new_centroid_map = partition(image, new_centroids, norm)
        new_cost = cost_function(new_centroids, new_centroid_map, norm)

        if new_cost < current_cost:
            centroids = new_centroids
            elements_per_centroid = new_centroid_map
            current_cost = new_cost
        else:
            break

        current_iteration += 1

    return centroids, elements_per_centroid


def python_generate_new_centroids(
    elements_per_centroid: np.ndarray
) -> np.ndarray:
    
    print("Generating new centroids")
    new_centroids = []
    for i in range(elements_per_centroid.shape[0]):
        zero = np.zeros(3)
        for j in range(elements_per_centroid.shape[1]):
            zero += elements_per_centroid[i, j]
        new_centroids.append(zero / elements_per_centroid.shape[1])
    
    return np.array(new_centroids)
    

def generate_new_centroids(elements_per_centroid: list[np.ndarray]) -> np.ndarray:
    """
    Chooses new centroids based on the average distance of the points in the cluster to its current centroid
    
    :param elements_per_centroid: list of ndarrays of shape (x, 3) where x is the number of elements in the cluster.
    len(elements_per_centroid) is the number of centroids

    :return: np.ndarray of shape (k, 3), the new centroids are in the same order as the input centroids
    """
    new_centroids = [None] * len(elements_per_centroid)
    
    for i, cluster in enumerate(elements_per_centroid):
        new_centroids[i] = np.mean(cluster, axis=0)
        
    return np.array(new_centroids)
    
    


def partition(image: np.ndarray, centroids: np.ndarray, norm) -> list[np.ndarray]:
    """
    Assign each pixel in the image to the nearest centroid.
    
    param: image: np.ndarray of shape (m, n, 3), the input image
    
    param: centroids: np.ndarray of shape (k, 3), where k is the number of centroids
    
    param: norm: the norm to use to calculate the distance between the pixel and the centroid
    
    returns: list of np.ndarray of shape (x, 3) where x is the number of elements in the cluster.
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
    min_norms = np.argmin(norms, axis=1)    
    
    # Create a map of the centroids to the pixels, this is done in a pythonic way
    centroids_map = [[] for _ in range(k)]
    for i in range(m * n):
        centroids_map[min_norms[i]].append(flat_image[i])
        
    
    ans =  [np.array(cluster) for cluster in centroids_map]
        
        
    return ans

    

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
    centroids: np.ndarray, elements_per_centroid: list[np.ndarray], norm
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
        norm_diff = np.linalg.norm(elements_per_centroid[i] - centroids[i][np.newaxis], norm, axis=1)
        cost += np.sum(norm_diff)
        
    return cost
    
    

