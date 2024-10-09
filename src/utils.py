import cv2
import numpy as np
import os
import time
import functools


def performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # print(f"Function {func.__name__} took {end - start} seconds")
        return result

    return wrapper


# function to open image
def open_image_from_path(image_path: str):
    image = cv2.imread(image_path)
    return image



# function to display image
def display_image(image: np.ndarray, window_name: str, resize=False) -> None:
    # if resize then resize the image to hight 500 and width to the same aspect ratio

    if not resize:
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
    extension = image_path.split("/")[-1].split(".")[1]
    for arg in args:
        new_name += f"_{arg}"
    return new_name +  "." + extension


def log(func, message=None):
    print(f"From {func.__name__}: {message}")

