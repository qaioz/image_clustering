import numpy as np
from collections import Counter


def partition(
    colors: np.ndarray, clusters: np.ndarray, norm: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partition the colors into k clusters based on the closest centroid.

    :param colors: np.ndarray of shape (n, 3)

    :param clusters: np.ndarray of shape (k, 3), where k is the number of clusters

    :param norm: Acts like ord parameter of np.linalg.norm function.

    :return nearest_centroid_indices: np.ndarray of shape (n,), the index of the closest centroid for each color
    I.e nearest_cluster_indices[i] = j => the ith color is closest to clusters[j]
    """

    # Calculate the distance between each unique color and each centroid
    # unique_colors[:, np.newaxis, :] expands unique_colors to shape (n, 1, 3) to allow broadcasting with centroids
    distances = np.linalg.norm(colors[:, np.newaxis, :] - clusters, axis=2, ord=norm)

    # Get the index of the closest centroid for each color
    nearest_cluster_indices = np.argmin(distances, axis=1)

    return nearest_cluster_indices


def select_clusters(image: np.ndarray, k: int) -> np.ndarray:
    """
    Select k random centroids from the image.

    each centroid is a color rgb which is of a  shape (3,), so the return value is a np.ndarray of shape (k, 3)

    This function is very cheap, so I will not use numpy vectorization
    """

    m, n, _ = image.shape
    clusters = []

    for _ in range(k):
        x = np.random.randint(0, m)
        y = np.random.randint(0, n)
        while tuple(image[x, y]) in clusters:
            x = np.random.randint(0, m)
            y = np.random.randint(0, n)

        clusters.append(tuple(image[x, y]))

    return np.array(clusters)


def cost_function(
    colors: np.ndarray,
    clusters: np.ndarray,
    color_cluster_indices: np.ndarray,
    color_frequencies: np.ndarray,
    norm: float,
) -> float:

    """
    Compute the cost function for the k-medoids algorithm.

    :param colors: np.ndarray of shape (n, 3), the unique colors in the image

    :param clusters: np.ndarray of shape (k, 3), the cluster medoids

    :param color_cluster_indices: np.ndarray of shape (n,), the index of the closest centroid for each color

    :param color_frequencies: np.ndarray of shape (n,), the frequency of each color in the image

    :param norm: float, the norm to use for the distance calculation
    """

    color_clusters = get_color_clusters(colors, clusters, color_cluster_indices)

    # Calculate the distance between color and centroid using the specified norm
    distances = np.linalg.norm(
        color_clusters[:, 0] - color_clusters[:, 1], axis=1, ord=norm
    )

    # Calculate the weighted sum of distances using the color frequencies
    total_cost = np.sum(distances * color_frequencies)

    return total_cost


def get_image_unique_colors_and_frequencies(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the unique colors in the image and their frequencies efficiently.

    :param image: np.ndarray of shape (m, n, 3)

    :return: tuple[unique_colors, frequencies], where unique_colors is an np.ndarray of shape (p, 3) containing the unique colors,
    and frequencies[i] == x means that unique_colors[i] appears x times in the image.
    """

    # Flatten the image into a 2D array of RGB values
    flat_image = image.reshape(-1, 3)

    # Convert RGB colors to single integers for faster counting
    rgb_as_ints = np.dot(flat_image, [1, 256, 65536])  # Weights for R, G, B components

    # Count occurrences of each unique integer using a Counter
    color_counter = Counter(rgb_as_ints)

    # Get unique RGB colors by extracting keys and converting back to (R, G, B)
    unique_colors_int = np.array(list(color_counter.keys()))
    unique_colors = np.stack(
        [
            (unique_colors_int >> 0) & 255,  # Blue component
            (unique_colors_int >> 8) & 255,  # Green component
            (unique_colors_int >> 16) & 255,  # Red component
        ],
        axis=1,
    )

    # Get frequencies from the Counter
    color_frequencies = np.array(list(color_counter.values()))

    return unique_colors, color_frequencies


def get_new_image_from_original_image_and_clusters(
    image: np.ndarray, clusters: np.ndarray, norm: float, chunk_size: int = 500
):
    """
    Create a new image from the original image and the cluster medoids, processing in chunks.

    :param image: np.ndarray of shape (m, n, 3), the original image
    :param clusters: np.ndarray of shape (k, 3)
    :param chunk_size: int, the number of rows to process at a time, before introducnig this, there was a memory overflows for large images

    :return: np.ndarray of shape (m, n, 3), the new image
    """

    # Get the dimensions of the image
    h, w, c = image.shape

    # Create an empty array to hold the new image
    new_image = np.zeros_like(image)

    # Process the image in chunks (by rows)
    for start_row in range(0, h, chunk_size):
        end_row = min(start_row + chunk_size, h)

        # Extract a chunk (rows from start_row to end_row)
        chunk = image[start_row:end_row, :, :].reshape(-1, 3)

        # Apply the clustering to the chunk
        color_cluster_indices = partition(chunk, clusters, norm)

        # Get the new colors for the chunk
        new_chunk = clusters[color_cluster_indices]

        # Reshape and place the chunk back in the new image
        new_image[start_row:end_row, :, :] = new_chunk.reshape(
            (end_row - start_row, w, c)
        )

    return new_image


def generate_new_clusters(
    colors: np.ndarray,
    color_frequencies: np.ndarray,
    clusters: np.ndarray,
    color_cluster_indices: np.array,
) -> np.ndarray:
    """
    This function is only for kmeans, It is not used in kmedoids.

    Generate new clusters based on the sum of each color in the cluster divided by the number of colors in the cluster

    :param unique_colors: np.ndarray of shape (n, 3), where n is the number of unique colors in the image

    :param color_frequencies: np.ndarray of shape (n,), the frequency of each color in the image

    :param color_centroid_indices: np.array of shape (n,), the index of the closest centroid for each color

    :param centroids: np.ndarray of shape (k, 3), where k is the number of centroids

    :return: np.ndarray of shape (k, 3) containing the new clusters
    """

    new_clusters = []

    for i in range(clusters.shape[0]):
        # get the colors in the cluster
        cluster_colors = colors[color_cluster_indices == i]

        # get the frequencies of the colors in the cluster
        cluster_frequencies = color_frequencies[color_cluster_indices == i]

        # calculate the new centroid
        new_cluster = np.sum(
            cluster_colors * cluster_frequencies[:, None], axis=0
        ) / np.sum(cluster_frequencies)

        new_clusters.append(new_cluster)

    return np.array(new_clusters)


def get_color_clusters(
    colors: np.ndarray, clusters: np.ndarray, color_cluster_indices: np.ndarray
) -> np.ndarray:
    """
    This is a helper function for cost_function. And is created soley to make the tests easier to write.

    Get the color centroids for each color in the image

    :param colors: np.ndarray of shape (n, 3) - an array of RGB colors

    :param centroids: np.ndarray of shape (k, 3) - an array of RGB centroid values

    :param color_centroid_indices: np.ndarray of shape (n,) - the index of the closest centroid for each color

    :returns: np.ndarray of shape (n, 2, 3), where each element contains the color and its corresponding centroid
    """

    # Initialize an empty array to store the colors and their centroids
    color_clusters = np.zeros((colors.shape[0], 2, 3))

    # Fill in the colors
    color_clusters[:, 0, :] = colors

    # Fill in the corresponding centroids
    color_clusters[:, 1, :] = clusters[color_cluster_indices]

    return color_clusters
