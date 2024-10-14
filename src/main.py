import logging
from src.compression.image_compression import compress_image, decompress_image
import os
from enum import Enum
from src.clustering.image_clustering import kmeans, kmedoids

logging.basicConfig(level=logging.INFO, format = "%(asctime)s - %(message)s")

COMPRESSED_DIR = "output/compressed_files"
DECOMPRESSED_DIR = "output/decompressed_files"
CLUSTEREED_DIR = "output/clustered_images"
INPUT_IMAGES_DIR = "input_images"
# create these directories if they don't exist
os.makedirs(CLUSTEREED_DIR, exist_ok=True)
os.makedirs(DECOMPRESSED_DIR, exist_ok=True)
os.makedirs(CLUSTEREED_DIR, exist_ok=True)

COMPRESSED_FILE_EXTENSION = ".gcp" # GCP stands for Gaioz Compressed Picture
DEFAULT_IMAGE_EXTENSION = ".bmp"

class Compression_Algorithm(Enum):
    KMEANS = 1
    KMEDOIDS = 2
    


def compress_image_main(image_path: str, *, algorithm: Compression_Algorithm = Compression_Algorithm.KMEANS, n_of_clusters: int = 10, iterations: int = 10, norm: float = 2) -> str:
    logging.info(f"\n Compressing image {image_path} using {algorithm.name} algorithm with {n_of_clusters} clusters, {iterations} iterations and norm {norm}")
    logging.info("Original image size: {:.2f} KB".format(os.path.getsize(image_path) / 1024))
    if algorithm == Compression_Algorithm.KMEANS:
        algorithm = lambda image: kmeans(image, num_clusters=n_of_clusters, max_iterations=iterations, norm=norm)
    else:
        algorithm = lambda image: kmedoids(image, n_clusters=n_of_clusters, max_iterations=iterations, norm=norm)
    
    output_file = _generate_compressed_file_path(image_path, algorithm, n_of_clusters, iterations, norm)
    compress_image(image_path, algorithm=algorithm, output_file=output_file)
    logging.info(f"Compression completed, the compressed file is saved at {output_file}")
    logging.info("Compressed image size: {:.2f} KB".format(os.path.getsize(output_file) / 1024))
    
    return output_file
    

def decompress_image_main(compressed_file_path: str) -> str:
    logging.info(f"\n Decompressing image {compressed_file_path}")
    logging.info("Compressed image size: {:.2f} KB".format(os.path.getsize(compressed_file_path) / 1024))
    
    output_file = _generate_decompressed_file_path(compressed_file_path)
    decompress_image(compressed_file_path, output_file)
    
    logging.info(f"Decompression completed, the decompressed file is saved at {output_file}")
    logging.info("Decompressed image size: {:.2f} KB".format(os.path.getsize(output_file) / 1024))
    return output_file
    



def _generate_compressed_file_path(image_path: str, algorithm: Compression_Algorithm, n_of_clusters: int, iterations: int, norm: float) -> str:
    """
    Get the name of the compressed image file and add keyword information to it.
    """
    
    filename = image_path.split("/")[-1].split(".")[0]
    
    algorithm_name = "kmeans" if algorithm == Compression_Algorithm.KMEANS else "kmedoids"
    
    # keywords
    kwargs = {
        "algorithm": algorithm_name,
        "clusters": n_of_clusters,
        "iterations": iterations,
        "norm": norm
    }
    
    # generate the new name build path from COMPRESSED_DIR
    new_name = filename
    for key, value in kwargs.items():
        new_name += f"_{key}={value}"
    
    new_name += COMPRESSED_FILE_EXTENSION
    
    return os.path.join(COMPRESSED_DIR, new_name)
    
    
def _generate_decompressed_file_path(compressed_file_path: str) -> str:
    """
    Get the name of the decompressed image file.
    """
    
    filename = compressed_file_path.split("/")[-1].split(".")[0]
    
    return os.path.join(DECOMPRESSED_DIR, filename + DEFAULT_IMAGE_EXTENSION)


if __name__ == "__main__":
    img = "input_images/sample1.bmp"
    compressed_file = compress_image_main(img, n_of_clusters=2)
    decomressed_file = decompress_image_main(compressed_file)
    
