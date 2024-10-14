import cv2
import numpy as np
import os
import time
import functools


# I want performance counter that remmembers the first 10 times and make available the map
# of the function name and the average time it took to run the function
def performance(func):
    # create a dictionary to store the function name and the list of times the function was called
    func_times = {}

    # function to calculate the average time
    average_time = lambda fun_name: sum(func_times[fun_name]) / len(
        func_times[fun_name]
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # get the function name
        fun_name = func.__name__
        # get the current time
        start = time.perf_counter()
        # run the function
        result = func(*args, **kwargs)
        # get the end time
        end = time.perf_counter()
        # calculate the time it took to run the function
        time_taken = end - start
        # if the function name is not in the dictionary, add it
        if fun_name not in func_times:
            func_times[fun_name] = []
        # append the time it took to run the function to the list
        func_times[fun_name].append(time_taken)
        # return the result
        return result

    # add the average time function to the wrapper
    wrapper.average_time = lambda: {
        fun_name: average_time(fun_name) for fun_name in func_times
    }
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
    return new_name + "." + extension


def log(func, message=None):
    print(f"From {func.__name__}: {message}")
