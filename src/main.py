from kmeans import kmeans
from utils import (
    open_image_from_path,
    convert_centroids_and_point_clusters_to_image,
    display_image,
    count_numer_of_different_colors,
    save_image,
    generate_new_name,
)
from src.kmedoids import kmedoids


def main():

    image_path = "images/Good Quality Wallpaper High Desktop.jpg"
    # image_path = "images/grass.png"
    # image_path = "images/zebra.webp"
    image = open_image_from_path(image_path)
    n_of_colors = count_numer_of_different_colors(image)
    print(f"Number of colors in original image: {n_of_colors}")
    print(f"Number of pixels in original image: {image.shape[0] * image.shape[1]}")
    print(f"npixels/ncolors: {image.shape[0] * image.shape[1] / n_of_colors}")
    k = 5
    max_iterations = 5
    norm = 1

    centroids, point_clusters = kmeans(
        image, num_clusters=k, max_iterations=max_iterations, norm=norm
    )
    new_image = convert_centroids_and_point_clusters_to_image(
        image.shape, centroids, point_clusters
    )

    new_image_name = generate_new_name(image_path, k, max_iterations, norm)
    save_image(new_image, new_image_name)
    display_image(new_image, new_image_name, resize=True)

    # display_image(new_image, "new_image", resize=False)


def medoids():

    image_path = "generated_images/zebra_100_3_2.png"
    image = open_image_from_path(image_path)
    n_of_colors = count_numer_of_different_colors(image)
    print(f"Number of colors in original image: {n_of_colors}")
    print(f"Number of pixels in original image: {image.shape[0] * image.shape[1]}")
    print(f"npixels/ncolors: {image.shape[0] * image.shape[1] / n_of_colors}")
    k = 3
    max_iterations = 1000
    norm = 2

    new_image = kmedoids(image, n_clusters=k, max_iter=max_iterations, norm=norm)

    new_image_name = generate_new_name(image_path, k, max_iterations, norm)
    save_image(new_image, new_image_name)
    display_image(new_image, new_image_name, resize=True)

if __name__ == "__main__":
    print("Running main")
    medoids()
    # main()
