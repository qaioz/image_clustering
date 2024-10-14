import logging
from src.compression.image_compression import compress_image, decompress_image
import os
from src.clustering.image_clustering import kmeans, kmedoids
from src.constants import (
    Compression_Algorithm,
    DECOMPRESSED_DIR,
    CLUSTEREED_DIR,
    INPUT_IMAGES_DIR
)
from src.utils import generate_compressed_file_path, generate_decompressed_file_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


os.makedirs(CLUSTEREED_DIR, exist_ok=True)
os.makedirs(DECOMPRESSED_DIR, exist_ok=True)
os.makedirs(CLUSTEREED_DIR, exist_ok=True)



def run_compress_image(
    image_path: str,
    *,
    algorithm: Compression_Algorithm = Compression_Algorithm.KMEANS,
    n_of_clusters: int = 10,
    iterations: int = 10,
    norm: float = 2,
    output_file: str | None = None,
) -> str:
    logging.info(
        f"\n Compressing image {image_path} using {algorithm.name} algorithm with {n_of_clusters} clusters, {iterations} iterations and norm {norm}"
    )
    logging.info(
        "Original image size: {:.2f} KB".format(os.path.getsize(image_path) / 1024)
    )
    if algorithm == Compression_Algorithm.KMEANS:
        algorithm = lambda image: kmeans(
            image, num_clusters=n_of_clusters, max_iterations=iterations, norm=norm
        )
    else:
        algorithm = lambda image: kmedoids(
            image, n_clusters=n_of_clusters, max_iterations=iterations, norm=norm
        )

    if output_file is None:
        output_file = generate_compressed_file_path(
            image_path, algorithm, n_of_clusters, iterations, norm
        )
    compress_image(image_path, algorithm=algorithm, output_file=output_file)
    logging.info(
        f"Compression completed, the compressed file is saved at {output_file}"
    )
    logging.info(
        "Compressed image size: {:.2f} KB".format(os.path.getsize(output_file) / 1024)
    )

    return output_file


def run_decompress_image(compressed_file_path: str) -> str:
    logging.info(f"\n Decompressing image {compressed_file_path}")
    logging.info(
        "Compressed image size: {:.2f} KB".format(
            os.path.getsize(compressed_file_path) / 1024
        )
    )

    output_file = generate_decompressed_file_path(compressed_file_path)
    decompress_image(compressed_file_path, output_file)

    logging.info(
        f"Decompression completed, the decompressed file is saved at {output_file}"
    )
    logging.info(
        "Decompressed image size: {:.2f} KB".format(os.path.getsize(output_file) / 1024)
    )
    return output_file


if __name__ == "__main__":
    img = INPUT_IMAGES_DIR + "/sample2.bmp"
    compressed_file = run_compress_image(img)
    decomressed_file = run_decompress_image(compressed_file)
