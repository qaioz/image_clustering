import numpy as np


def partition(
    unique_colors: np.ndarray, centroids: np.ndarray, norm: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partition the colors into k clusters based on the closest centroid.

    :param unique_colors: np.ndarray of shape (n, 3), where n is the number of unique colors in the image

    :param centroids: np.ndarray of shape (k, 3), where k is the number of centroids

    :param norm: Acts like ord parameter of np.linalg.norm function. If None, the Euclidean norm is used.

    :return nearest_centroid_indices: np.ndarray of shape (n,), the index of the closest centroid for each color
    """

    # Calculate the distance between each unique color and each centroid
    # unique_colors[:, np.newaxis, :] expands unique_colors to shape (n, 1, 3) to allow broadcasting with centroids
    distances = np.linalg.norm(
        unique_colors[:, np.newaxis, :] - centroids, axis=2, ord=norm
    )

    # Get the index of the closest centroid for each color
    nearest_centroid_indices = np.argmin(distances, axis=1)

    return nearest_centroid_indices


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
    color_centroids: np.ndarray, color_frequencies: np.ndarray, norm: float | None
) -> float:

    """
    Compute the cost function for the k-medoids algorithm.

    :param color_centroids: np.ndarray of shape (k, 2, 3). cc \in color_centroids, cc = [color, centroid],
                            where color and centroid are np.ndarrays of shape (3,)
    :param color_frequencies: np.ndarray of shape (k,).  cf \in color_frequencies,
                              cf = how many times the color appears in the image
    :param norm: Acts like ord parameter of np.linalg.norm function. If None, the Euclidean norm is used.

    :return: float
    """

    # Calculate the distance between color and centroid using the specified norm
    distances = np.linalg.norm(
        color_centroids[:, 0] - color_centroids[:, 1], axis=1, ord=norm
    )

    # Calculate the weighted sum of distances using the color frequencies
    total_cost = np.sum(distances * color_frequencies)

    return total_cost

def get_image_unique_colors_and_frequencies(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the unique colors in the image and their frequencies.

    :param image: np.ndarray of shape (m, n, 3)

    :return: tuple[np.ndarray, np.ndarray]
    """

    # Get the unique colors and their frequencies
    unique_colors, color_frequencies = np.unique(
        image.reshape(-1, 3), axis=0, return_counts=True
    )

    return unique_colors, color_frequencies

def get_new_image_from_original_image_and_medoids(
    image: np.ndarray, medoids: np.ndarray
):
    """
    Create a new image from the original image and the cluster medoids.

    :param image: np.ndarray of shape (m, n, 3), the original image

    :param medoids: np.ndarray of shape (k, 3), the medoids of the clusters

    :return: np.ndarray of shape (m, n, 3), the new image
    """

    # flatten the image
    flat_image = image.reshape(-1, 3)

    # use partition to get the color centroids
    color_centroid_indices = partition(flat_image, medoids, None)

    # get the centroids for each color
    new_image = medoids[color_centroid_indices]

    # reshape the new image to the original shape
    new_image = new_image.reshape(image.shape)
    
    return new_image

def generate_new_clusters(
    unique_colors: np.ndarray, color_frequencies: np.ndarray, color_centroid_indices: np.array, centroids: np.ndarray,
) -> np.ndarray:
    """
    Generate new clusters based on the sum of each color in the cluster divided by the number of colors in the cluster

    :param unique_colors: np.ndarray of shape (n, 3), where n is the number of unique colors in the image
    
    :param color_frequencies: np.ndarray of shape (n,), the frequency of each color in the image

    :param color_centroid_indices: np.array of shape (n,), the index of the closest centroid for each color

    :param centroids: np.ndarray of shape (k, 3), where k is the number of centroids

    :return: np.ndarray of shape (k, 3) containing the new clusters
    """

    new_centroids = []

    for i in range(centroids.shape[0]):
        # get the colors in the cluster
        cluster_colors = unique_colors[color_centroid_indices == i]

        # get the frequencies of the colors in the cluster
        cluster_frequencies = color_frequencies[color_centroid_indices == i]

        # calculate the new centroid
        new_centroid = np.sum(cluster_colors * cluster_frequencies[:, None], axis=0) / np.sum(cluster_frequencies)

        new_centroids.append(new_centroid)
        
    return np.array(new_centroids)

def get_color_centroids(
    colors: np.ndarray, centroids: np.ndarray, color_centroid_indices: np.ndarray
) -> np.ndarray:
    """
    Get the color centroids for each color in the image

    :param colors: np.ndarray of shape (n, 3) - an array of RGB colors

    :param centroids: np.ndarray of shape (k, 3) - an array of RGB centroid values

    :param color_centroid_indices: np.ndarray of shape (n,) - the index of the closest centroid for each color

    :returns: np.ndarray of shape (n, 2, 3), where each element contains the color and its corresponding centroid
    """

    # Initialize an empty array to store the colors and their centroids
    color_centroids = np.zeros((colors.shape[0], 2, 3))

    # Fill in the colors
    color_centroids[:, 0, :] = colors

    # Fill in the corresponding centroids
    color_centroids[:, 1, :] = centroids[color_centroid_indices]

    return color_centroids
