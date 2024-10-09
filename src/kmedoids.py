import numpy as np
from src.utils import log


def kmedoids(
    image: np.ndarray,
    n_clusters: int,
    max_iter: int,
    norm: float | None = None,
    threshold: float = 1e-3
) -> np.ndarray:
    """
    Function to perform the k-medoids algorithm on the image.

    :param image: np.ndarray of shape (m, n, 3), the input image
    
    :param n_clusters: int, the number of clusters
    
    :param max_iter: int, the maximum number of iterations
    
    :param norm: works as the argument 'ord' in numpy.linalg.norm function
    
    :param threshold: float, cost threshold. If the cost improvement is smaller than the threshold, the algorithm stops.
    
    :return: np.ndarray of shape (m, n, 3) containing the image with the k-medoids applied
    """

    # Get the unique colors in the image and count their frequencies
    unique_colors, color_frequencies = get_image_unique_colors_and_frequencies(image)
    
    # Step 1: Randomly select initial medoids
    centroids = select_centroids(image, n_clusters)
    
    # Step 2: Partition the colors based on the closest centroid and calculate the cost
    color_centroids = partition(unique_colors, centroids, norm)
    current_cost = cost_function(color_centroids, color_frequencies, norm)
    
    # Iterative optimization loop
    for iteration in range(max_iter):

        # Step 3: Try swapping medoids with non-medoids to find a lower cost configuration
        for i in range(n_clusters):
            for j in range(unique_colors.shape[0]):
                # If the color is already a medoid, skip it
                if any(np.array_equal(unique_colors[j], c) for c in centroids):
                    continue

                # Swap medoid i with non-medoid color j
                new_centroids = centroids.copy()
                new_centroids[i] = unique_colors[j]

                # Repartition and compute the cost of the new medoid configuration
                new_color_centroids = partition(unique_colors, new_centroids, norm)
                new_cost = cost_function(new_color_centroids, color_frequencies, norm)

                # Check if the new cost is lower than the current cost
                if new_cost < current_cost:
                    centroids = new_centroids
                    current_cost = new_cost
                

        log(kmedoids, f"Iteration {iteration}, current cost: {current_cost}")
        
    return get_new_image_from_original_image_and_medoids(image, centroids)



def partition(
    unique_colors: np.ndarray, centroids: np.ndarray, norm: float | None
) -> np.ndarray:
    """
    Partition the colors into k clusters based on the closest centroid.

    :param unique_colors: np.ndarray of shape (n, 3), where n is the number of unique colors in the image

    :param centroids: np.ndarray of shape (k, 3), where k is the number of centroids

    :param norm: Acts like ord parameter of np.linalg.norm function. If None, the Euclidean norm is used.

    :return color_centroids: np.ndarray of shape (n, 2, 3). cc \in color_centroids, cc = [color, centroid],
             where color and centroid are np.ndarrays of shape (3,)
    """

    # Calculate the distance between each unique color and each centroid
    # unique_colors[:, np.newaxis, :] expands unique_colors to shape (n, 1, 3) to allow broadcasting with centroids
    distances = np.linalg.norm(
        unique_colors[:, np.newaxis, :] - centroids, axis=2, ord=norm
    )

    # Get the index of the closest centroid for each color
    nearest_centroid_indices = np.argmin(distances, axis=1)

    # Construct the color_centroids array by stacking unique_colors and their closest centroids
    color_centroids = np.empty(
        (unique_colors.shape[0], 2, 3), dtype=unique_colors.dtype
    )
    color_centroids[:, 0, :] = unique_colors
    color_centroids[:, 1, :] = centroids[nearest_centroid_indices]

    return color_centroids


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


def get_new_image_from_original_image_and_medoids(image: np.ndarray, medoids: np.ndarray):
    """
    Create a new image from the original image and the cluster medoids.

    :param image: np.ndarray of shape (m, n, 3), the original image

    :param medoids: np.ndarray of shape (k, 3), the medoids of the clusters

    :return: np.ndarray of shape (m, n, 3), the new image
    """

    # flatten the image
    flat_image = image.reshape(-1, 3)
    
    # use partition to get the color centroids
    color_centroids = partition(flat_image, medoids, None)
    
    # leave only the centroids
    new_flatted_image = color_centroids[:, 1, :]
    
    # reshape the new image
    new_image = new_flatted_image.reshape(image.shape)
    
    return new_image
    