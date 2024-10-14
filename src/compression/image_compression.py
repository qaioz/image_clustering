import numpy as np
from src.clustering.image_clustering import kmeans, kmedoids
from enum import Enum
from src.utils import open_image_from_path


# enum compression_algorithm
class Compression_Algorithm(Enum):
    KMEANS = 1
    KMEDOIDS = 2



def compress_image(image_path: str, *, algorithm: callable, output_file: str) -> None:
    
    image = open_image_from_path(image_path)
    
    # Perform clustering on the image
    clusters, clustered_image = algorithm(image)

    # Compress the clustered image
    compressed = compress_clustered_image(clustered_image, clusters)

    # Save the compressed image to a file
    _save_compressed_image_binary(
        compressed_image=compressed,
        original_dimensions=image.shape[:2],
        clusters=clusters,
        output_file=output_file,
    )
    
    return





def compress_clustered_image(image: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Compress an already clustered image using the given clusters, counting how many times
    each centroid repeats consecutively (linear compression) using only NumPy.

    Parameters:
    - image (ndarray): Already clustered input image as an n x m x 3 ndarray (height x width x color channels).
    - clusters (ndarray): Clusters as a k x 3 ndarray (number of clusters x color channels).

    Returns:
    - compressed (ndarray): An (m*n, 2) ndarray where the first column is the centroid index
      and the second column is the count of consecutive repetitions.
    """
    # Flatten the image to shape (n*m, 3)
    flattened_image = image.reshape(-1, 3)

    # Calculate indices of the exact cluster for each pixel by comparing the differences
    distances = np.linalg.norm(flattened_image[:, None, :] - clusters[None, :, :], axis=2)
    centroid_indices = np.argmin(distances, axis=1)

    # Find where the centroid indices change (from one centroid to another)
    change_points = np.where(centroid_indices[:-1] != centroid_indices[1:])[0] + 1

    # Include the start and end of the array as change points
    change_points = np.concatenate(([0], change_points, [len(centroid_indices)]))

    # Compute the lengths of consecutive runs of each centroid
    run_lengths = np.diff(change_points)

    # Get the centroid values at the change points
    centroids_at_change = centroid_indices[change_points[:-1]]

    # Stack the centroids and run lengths to form the compressed result
    compressed = np.column_stack((centroids_at_change, run_lengths))

    return compressed


def _save_compressed_image_binary(
    compressed_image: np.ndarray, 
    *,
    original_dimensions: tuple[int, int], 
    clusters: np.ndarray, 
    output_file: str
) -> None:
    """
    Save the compressed image to a binary file.
    """
    height, width = original_dimensions
    num_clusters = len(clusters)
    
    # Open the specified file to write the compressed image in binary format
    with open(output_file, "wb") as file:
        # Write dimensions (height, width) and number of clusters
        file.write(height.to_bytes(2, byteorder='big'))
        file.write(width.to_bytes(2, byteorder='big'))
        file.write(num_clusters.to_bytes(1, byteorder='big'))
        
        # Write cluster colors
        for cluster in clusters:
            file.write(bytearray(cluster))
        
        # Write compressed data (cluster index and counts)
        for row in compressed_image:
            row = int(row[0]), int(row[1])
            file.write(row[0].to_bytes(1, byteorder='big'))
            file.write(row[1].to_bytes(3, byteorder='big'))
            
            
