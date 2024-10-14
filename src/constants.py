from enum import Enum   


# enum compression_algorithm
class Compression_Algorithm(Enum):
    KMEANS = 1
    KMEDOIDS = 2
    
class ClusteringAlgorithm(Enum):
    KMEANS = 1
    KMEDOIDS = 2

COMPRESSED_DIR = "output/compressed_files"
DECOMPRESSED_DIR = "output/decompressed_files"
CLUSTEREED_DIR = "output/clustered_images"
INPUT_IMAGES_DIR = "input_images"


COMPRESSED_FILE_EXTENSION = ".gcp"  # GCP stands for Gaioz Compressed Picture
DEFAULT_IMAGE_EXTENSION = ".bmp"


