import numpy as np
from src.utils import log
from src.commons import (
    select_centroids,
    cost_function,
    partition,
    get_image_unique_colors_and_frequencies,
    get_new_image_from_original_image_and_medoids,
    generate_new_clusters,
    get_color_centroids
)


def kmeans(
    image: np.ndarray,
    *,
    num_clusters: int,
    max_iterations: int,
    norm: float | None = None,
    threshold: float = 1e-3,
) -> np.ndarray:
    """
    kmeans clustering algorithm

    :param image: np.ndarray of shape (m, n, 3), the input image

    :param num_clusters: int, the number of clusters

    :param max_iterations: int, the maximum number of iterations

    :param norm: works as the argument 'ord' in numpy.linalg.norm function

    :param threshold: float, cost threshold. if new_cost >= current_cost - threshold, then the algorithm stops. I.e.
    If the cost of new clusters is less than the cost of the current clusters minus the threshold, the new clusters will be used, otherwise the algorithm iteration will stop.

    :return: np.ndarray of shape (m, n, 3) containing the image with the k-means applied
    """

    unique_colors, color_frequencies = get_image_unique_colors_and_frequencies(image)

    centroids = select_centroids(image, num_clusters)

    color_centroid_indices = partition(unique_colors, centroids, norm)
    
    current_cost = cost_function(get_color_centroids(unique_colors,centroids, color_centroid_indices), color_frequencies, norm)    
    
    log(kmeans, f"initial centroids: {centroids}")

    for iteration in range(2,max_iterations+1):

        new_centroids = generate_new_clusters(
            centroids=centroids,
            unique_colors=unique_colors,
            color_frequencies=color_frequencies,
            color_centroid_indices=color_centroid_indices,
        )

        new_color_centroid_indices = partition(unique_colors, new_centroids, norm)
        new_cost = cost_function(get_color_centroids(unique_colors, new_centroids, color_centroid_indices), color_frequencies, norm)


        if new_cost >= current_cost - threshold:
            print(f"cost is not improving, stopping at iteration {iteration}")
            break
        else:
            centroids = new_centroids
            color_centroid_indices = new_color_centroid_indices
            current_cost = new_cost
        
        log(
            kmeans,
            f"iteration: {iteration}"
        )
        
    log(kmeans, f"final centroids: {centroids}")
    log(kmeans, f"final cost: {current_cost}")

    new_image = get_new_image_from_original_image_and_medoids(
        image=image, medoids=centroids, 
    )

    return new_image
