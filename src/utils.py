import cv2
import numpy as np
import os
from src.constants import (
    ClusteringAlgorithm,
)

from src.constants import (
    COMPRESSED_DIR,
    DECOMPRESSED_DIR,
    COMPRESSED_FILE_EXTENSION,
    DEFAULT_IMAGE_EXTENSION,
    CLUSTEREED_DIR,
    Compression_Algorithm
)

# function to open image
def open_image_from_path(image_path: str):
    image = cv2.imread(image_path)
    return image


def count_numer_of_different_colors(image: np.ndarray) -> int:
    """
    Count the number of different colors in the image

    param: image: np.ndarray of shape (m, n, 3)

    returns: int
    """

    return len(np.unique(image.reshape(-1, image.shape[2]), axis=0))



def generate_clustered_image_path(
    image_path: str,
    algorithm: ClusteringAlgorithm,
    n_of_clusters: int,
    iterations: int,
    norm: float,
) -> str:
    """
    Get the name of the clustered image file and add keyword information to it.
    """
    
    filename = image_path.split("/")[-1].split(".")[0]

    algorithm_name = (
        "kmeans" if algorithm == ClusteringAlgorithm.KMEANS else "kmedoids"
    )

    # keywords
    kwargs = {
        "algorithm": algorithm_name,
        "clusters": n_of_clusters,
        "iterations": iterations,
        "norm": norm,
    }

    # generate the new name build path from CLUSTERED_DIR
    new_name = filename
    for key, value in kwargs.items():
        new_name += f"_{key}={value}"

    new_name += DEFAULT_IMAGE_EXTENSION

    return os.path.join(CLUSTEREED_DIR, new_name)



def generate_compressed_file_path(
    image_path: str,
    algorithm: Compression_Algorithm,
    n_of_clusters: int,
    iterations: int,
    norm: float,
) -> str:
    """
    Get the name of the compressed image file and add keyword information to it.
    """

    filename = image_path.split("/")[-1].split(".")[0]

    algorithm_name = (
        "kmeans" if algorithm == Compression_Algorithm.KMEANS else "kmedoids"
    )

    # keywords
    kwargs = {
        "algorithm": algorithm_name,
        "clusters": n_of_clusters,
        "iterations": iterations,
        "norm": norm,
    }

    # generate the new name build path from COMPRESSED_DIR
    new_name = filename
    for key, value in kwargs.items():
        new_name += f"_{key}={value}"

    new_name += COMPRESSED_FILE_EXTENSION

    return os.path.join(COMPRESSED_DIR, new_name)


def generate_decompressed_file_path(compressed_file_path: str) -> str:
    """
    Get the name of the decompressed image file.
    """

    filename = compressed_file_path.split("/")[-1].split(".")[0]

    return os.path.join(DECOMPRESSED_DIR, filename + DEFAULT_IMAGE_EXTENSION)

