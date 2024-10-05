from kmeans import kmeans
from utils import (
    open_image_from_path,
    convert_centroids_and_point_clusters_to_image,
    display_image,
    count_numer_of_different_colors,
    save_image,
    generate_new_name,
)


def main():

    image_path = "images/Good Quality Wallpaper High Desktop.jpg"
    image_path = "images/grass.png"
    image_path = "images/zebra.webp"
    image = open_image_from_path(image_path)
    n_of_colors = count_numer_of_different_colors(image)
    print(f"Number of colors in original image: {n_of_colors}")
    k = 2
    max_iterations = 100
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


if __name__ == "__main__":
    print("Running main")
    main()
