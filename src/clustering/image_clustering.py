import numpy as np
from src.clustering.commons import (
    select_clusters,
    cost_function,
    partition,
    get_image_unique_colors_and_frequencies,
    get_new_image_from_original_image_and_clusters,
    generate_new_clusters,
)

COST_THRESHOLD = 1e-3


def kmeans(
    image: np.ndarray,
    *,
    num_clusters: int,
    max_iterations: int,
    norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    kmeans clustering algorithm

    :param image: np.ndarray of shape (m, n, 3), the input image

    :param num_clusters: int, the number of clusters

    :param max_iterations: int, the maximum number of iterations

    :param norm: works as the argument 'ord' in numpy.linalg.norm function

    :param threshold: float, cost threshold. if new_cost >= current_cost - threshold, then the algorithm stops. I.e.
    If the cost of new clusters is less than the cost of the current clusters minus the threshold, the new clusters will be used, otherwise the algorithm iteration will stop.

    :return: tuple(np.ndarray of shape (num_clusters, 3), np.ndarray of shape (m, n, 3)) containing the centroids and the clustered image
    """
    unique_colors, color_frequencies = get_image_unique_colors_and_frequencies(image)

    centroids = select_clusters(image, num_clusters)

    color_centroid_indices = partition(unique_colors, centroids, norm)

    current_cost = cost_function(
        unique_colors, centroids, color_centroid_indices, color_frequencies, norm
    )

    for iteration in range(2, max_iterations + 1):

        new_centroids = generate_new_clusters(
            clusters=centroids,
            colors=unique_colors,
            color_frequencies=color_frequencies,
            color_cluster_indices=color_centroid_indices,
        )

        new_color_centroid_indices = partition(unique_colors, new_centroids, norm)
        new_cost = cost_function(
            unique_colors,
            new_centroids,
            color_centroid_indices,
            color_frequencies,
            norm,
        )

        if new_cost >= current_cost - COST_THRESHOLD:
            break
        else:
            centroids = new_centroids
            color_centroid_indices = new_color_centroid_indices
            current_cost = new_cost

    new_image = get_new_image_from_original_image_and_clusters(
        image=image,
        clusters=centroids,
        norm=norm,
    )

    return centroids, new_image


def kmedoids(
    image: np.ndarray,
    *n_clusters: int,
    max_iterations: int,
    norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to perform the k-medoids algorithm on the image.

    :param image: np.ndarray of shape (m, n, 3), the input image

    :param n_clusters: int, the number of clusters

    :param max_iter: int, the maximum number of iterations

    :param norm: works as the argument 'ord' in numpy.linalg.norm function

    :param threshold: float, cost threshold. If the cost improvement is smaller than the threshold, the algorithm stops.

    :return: tuple centroids(k,3), image(m,n,3) containing the  centroids and the clustered image
    """

    # Get the unique colors in the image and count their frequencies
    unique_colors, color_frequencies = get_image_unique_colors_and_frequencies(image)

    # Step 1: Randomly select initial medoids
    centroids = select_clusters(image, n_clusters)

    # Step 2: Partition the colors based on the closest centroid and calculate the cost
    color_centroid_indices = partition(unique_colors, centroids, norm)
    current_cost = cost_function(
        unique_colors,
        centroids,
        color_centroid_indices,
        color_frequencies,
        norm,
    )

    # iterate until there are no swaps

    for iteration in range(2, max_iterations + 1):
        # for each medoid
        for i in range(n_clusters):
            # select 100 random colors
            for j in range(len(unique_colors)):
                # swap the medoid with the color
                new_centroids = centroids.copy()
                new_centroids[i] = unique_colors[j]

                # partition the colors based on the closest centroid and calculate the cost
                new_color_centroid_indices = partition(
                    unique_colors, new_centroids, norm
                )
                new_cost = cost_function(
                    unique_colors,
                    new_centroids,
                    new_color_centroid_indices,
                    color_frequencies,
                    norm,
                )

                # if the new cost is less than the current cost, update the medoids and the cost
                if new_cost < current_cost:
                    centroids = new_centroids
                    color_centroid_indices = new_color_centroid_indices
                    current_cost = new_cost

    new_image = get_new_image_from_original_image_and_clusters(
        image=image, clusters=centroids, norm=norm
    )
    return centroids, new_image
