import numpy as np
from src.utils import log
from src.commons import (
    get_image_unique_colors_and_frequencies,
    select_clusters,
    cost_function,
    partition,
    get_new_image_from_original_image_and_clusters,
    get_color_clusters
)


def kmedoids(
    image: np.ndarray,
    n_clusters: int,
    max_iter: int,
    norm: float | None = None,
    threshold: float = 1e-3,
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
    centroids = select_clusters(image, n_clusters)

    # Step 2: Partition the colors based on the closest centroid and calculate the cost
    color_centroid_indices = partition(unique_colors, centroids, norm)
    current_cost = cost_function(
        get_color_clusters(unique_colors, centroids, color_centroid_indices),
        color_frequencies,
        norm,
    )
    
    # iterate until there are no swaps
    
    for iteration in range(2, max_iter + 1):
        # for each medoid
        for i in range(n_clusters):
            # select 100 random colors
            for j in range(len(unique_colors)):
                # swap the medoid with the color
                new_centroids = centroids.copy()
                new_centroids[i] = unique_colors[j]
                
                # partition the colors based on the closest centroid and calculate the cost
                new_color_centroid_indices = partition(unique_colors, new_centroids, norm)
                new_cost = cost_function(
                    get_color_clusters(unique_colors, new_centroids, new_color_centroid_indices),
                    color_frequencies,
                    norm,
                )
                
                # if the new cost is less than the current cost, update the medoids and the cost
                if new_cost < current_cost:
                    centroids = new_centroids
                    color_centroid_indices = new_color_centroid_indices
                    current_cost = new_cost
                    print(f"found a better medoid at iteration {iteration}, medoid {i}, cost {current_cost}")
                    
            log(kmedoids, f"iteration: {iteration}, medoid: {i}, cost: {current_cost}")

    log(kmedoids, f"final centroids: {centroids}")
    
    new_image = get_new_image_from_original_image_and_clusters(
        image=image, clusters=centroids
    )
    return new_image
    