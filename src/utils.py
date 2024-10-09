import cv2
import numpy as np
import os


def convert_centroids_and_point_clusters_to_image(
    dimensions: tuple, centroids: np.ndarray, point_clusters: np.ndarray
) -> np.ndarray:
    """
    Convert the centroids and point clusters to an image

    param: centroids: np.ndarray of shape (k, 3)
    param: point_clusters: np.ndarray of shape (m, n)

    returns: np.ndarray of shape (m, n, 3)
    """

    image = np.zeros(dimensions, dtype=np.uint8)
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            image[i, j] = centroids[point_clusters[i, j]]

    return image


# function to open image
def open_image_from_path(image_path: str):
    image = cv2.imread(image_path)
    return image


def open_image_from_np_array(image: np.ndarray):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("opened image of shape: ", image_rgb.shape, "image type: ", type(image_rgb))
    return image_rgb


# function to display image
def display_image(image: np.ndarray, window_name: str, resize=False) -> None:
    # if resize then resize the image to hight 500 and width to the same aspect ratio

    if resize:
        image = resize_image(image, 500)

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(image: np.ndarray, new_width: int) -> np.ndarray:
    # height should be calculated based on the aspect ratio

    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(new_width / aspect_ratio)
    new_image = cv2.resize(image, (new_width, new_height))
    return new_image


def count_numer_of_different_colors(image: np.ndarray) -> int:
    """
    Count the number of different colors in the image

    param: image: np.ndarray of shape (m, n, 3)

    returns: int
    """

    return len(np.unique(image.reshape(-1, image.shape[2]), axis=0))


def save_image(image: np.ndarray, new_name: str) -> None:
    """
    Save the given image (as a numpy array) to a specified path under 'generated_images' folder using OpenCV.

    param: image: np.ndarray, the image to save.
    param: new_name: str, the new name (with extension) for the saved image file.
    """
    # Ensure the directory exists
    save_dir = "generated_images/"
    os.makedirs(save_dir, exist_ok=True)

    # Generate the full path
    save_path = os.path.join(save_dir, new_name)

    # Use OpenCV to save the image
    cv2.imwrite(save_path, image)

    print(f"Image saved at {save_path}")


def generate_new_name(image_path: str, *args) -> str:
    # generate a new name based on the image path and the arguments
    new_name = image_path.split("/")[-1].split(".")[0]
    for arg in args:
        new_name += f"_{arg}"
    return new_name + ".webp"


def log(func, message=None):
    print(f"From {func.__name__}: {message}")

