import numpy as np
from src.utils import (
    open_image_from_path,
    count_numer_of_different_colors,
    save_image,
    generate_new_name,
)
from src.clustering import kmeans, kmedoids


zebra_image_path = "images/zebra.webp"
grass_image_path = "images/grass.png"
iceberg_image_path = "images/iceberg.jpg"
small_grass_image_path = "images/small_grass.png"
grass_100_20_2_image_path = "images/grass_100_20_2.png"
telescopes_image_path = "images/telescopes.jpg"
elephant_image_path = "images/elephant.jpg"


def run_kmeans_and_display_image():
    image_path = iceberg_image_path

    image = open_image_from_path(image_path)
    # n_of_colors = count_numer_of_different_colors(image)
    # print(f"Number of colors in original image: {n_of_colors}")
    # print(f"Number of pixels in original image: {image.shape[0] * image.shape[1]}")
    # print(f"npixels/ncolors: {image.shape[0] * image.shape[1] / n_of_colors}")
    k = 4
    max_iterations = 40
    norm = 2

    new_image = kmeans(
        image, num_clusters=k, max_iterations=max_iterations, norm=norm
    )
    # print new image dimensions
    print("old image shape: ", image.shape)
    print("new image shape: ", new_image.shape)
    new_image_name = generate_new_name(image_path, k, max_iterations, norm)
    save_image(new_image, new_image_name)
    # display_image(new_image, new_image_name, resize=False)


def run_kmedoids_and_display_image():
    image_path = grass_100_20_2_image_path
    image = open_image_from_path(image_path)
    n_of_colors = count_numer_of_different_colors(image)
    print(f"Number of colors in original image: {n_of_colors}")
    print(f"Number of pixels in original image: {image.shape[0] * image.shape[1]}")
    print(f"npixels/ncolors: {image.shape[0] * image.shape[1] / n_of_colors}")
    k = 2
    max_iterations = 100
    norm = 1

    new_image = kmedoids(
        image, n_clusters=k, max_iter=max_iterations, norm=norm
    )

    new_image_name = generate_new_name(image_path, k, max_iterations, norm)
    save_image(new_image, "medoids" + new_image_name)
    # display_image(new_image, new_image_name, resize=False)


run_kmeans_and_display_image()
# run_kmedoids_and_display_image()


# v  = np.array([1,2,3])

# print(np.linalg.norm(v, -1))
