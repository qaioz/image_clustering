from kmeans import kmeans
from utils import (
    open_image_from_path,
    convert_cluster_map_to_image,
    display_image,
    resize_image,
    p_norm,
)

image_path = "images/Good Quality Wallpaper High Desktop.jpg"
image_path = "images/small_grass.png"


def main():
    image = open_image_from_path(image_path)
    centroids, cluster_map = kmeans(image, 3, 10, 2)
    new_image = convert_cluster_map_to_image(image, cluster_map)
    resized = resize_image(new_image, 500)
    display_image(resized, "KMeans")


if __name__ == "__main__":
    print("Running main")
    main()
