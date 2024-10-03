import cv2
import numpy as np
import time
from functools import wraps


def convert_cluster_map_to_image(   
    image: np.ndarray, cluster_map: dict[tuple, list[tuple]]
) -> np.ndarray:
    """
    Convert the cluster map to an image
    """

    new_image = np.zeros_like(image)

    for centroid, points in cluster_map.items():
        for point in points:
            new_image[point[0], point[1]] = np.array(centroid)

    return new_image


# function to open image
def open_image_from_path(image_path: str):
    image = cv2.imread(image_path)
    return image


def open_image_from_np_array(image: np.ndarray):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("opened image of shape: ", image_rgb.shape, "image type: ", type(image_rgb))
    return image_rgb


# function to display image
def display_image(image: np.ndarray, window_name: str) -> None:
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(image: np.ndarray, new_width: int) -> np.ndarray:
    # height should be calculated based on the aspect ratio

    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(new_width / aspect_ratio)
    new_image = cv2.resize(image, (new_width, new_height))
    return new_image


def p_norm(vector: np.array, p: int) -> float:
    """
    Calculate the p-norm of a vector

    param: vector: np.array
    param: p: int

    returns: float
    """

    return np.linalg.norm(vector, p)


def preformance_counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function {func.__name__} took {round(end - start, 3)} seconds")
        return result

    return wrapper
