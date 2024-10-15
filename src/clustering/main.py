from src.utils import open_image_from_path, generate_clustered_image_path, count_numer_of_different_colors
from src.clustering.image_clustering import kmeans, kmedoids
from src.constants import ClusteringAlgorithm
import logging
import cv2

from src.constants import INPUT_IMAGES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def run_clustering_and_generate_image(image_path: str, clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS, n_clusters: int = 10, max_iterations: int = 10, norm: float = 2, output_file: str | None = None) -> str:
    logging.info(f"\n Clustering image {image_path} using {clustering_algorithm.name} algorithm with {n_clusters} clusters, {max_iterations} iterations and norm {norm}")
    
    image_array = open_image_from_path(image_path)
    
    logging.info(f"\n Image Dimensions: {image_array.shape}")
    
    num_colors = count_numer_of_different_colors(image_array)
    logging.info(f"\n Number of different colors: {num_colors}")
    
    
    
    if clustering_algorithm == ClusteringAlgorithm.KMEANS:
        clusters, new_image = kmeans(image_array, num_clusters=n_clusters, max_iterations=max_iterations, norm=norm)
    else:
        clusters, new_image = kmedoids(image_array, n_clusters=n_clusters, max_iterations=max_iterations, norm=norm)
        
    # print final clusters
    
    logging.info(f"\n Final clusters: {clusters}")
    
    # save new image
    if not output_file:
        new_image_path = generate_clustered_image_path(image_path, clustering_algorithm, n_clusters, max_iterations, norm)
    else:
        new_image_path = output_file
        
    cv2.imwrite(new_image_path, new_image)
    logging.info(f"\n New image saved at {new_image_path}")
    return new_image_path
    


if __name__ == "__main__":
    
    image_path = INPUT_IMAGES_DIR + "/sample2.bmp"
    
    new_image_path = run_clustering_and_generate_image(image_path, clustering_algorithm=ClusteringAlgorithm.KMEANS, output_file=INPUT_IMAGES_DIR + "/sample2_kmeans10.bmp")
    
    new_image_path = run_clustering_and_generate_image(new_image_path, clustering_algorithm=ClusteringAlgorithm.KMEDOIDS, output_file=INPUT_IMAGES_DIR + "/kmedoidsonclustered.bmp")
    
