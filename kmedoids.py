import cv2
import numpy as np

image_path = "images/Good Quality Wallpaper High Desktop.jpg"
image_path = "images/small_grass.png"


def kmedoids(image: np.ndarray, k: int, norm_function) -> np.ndarray:
    """
    kmeans clustering algorithm

    param: image: np.ndarray of shape (m, n, 3)
    param: k: number of clusters
    param: norm_function: function to calculate the norm between two points

    """

    print("Running kmedoids")

    centroids = select_centroids(image, k)
    print("Selected medoids: ", centroids)
    centroid_map = partition(image, centroids, norm_function)
    costs = get_cost_per_medoid(image, centroid_map, norm_function)

    # for each medoid
    for medoid in centroids:
        print("processing medoid: ", medoid)
        # for each non-medoid point
        for point in centroid_map[medoid]:
            # temporarily swap the medoid with the point
            temp_medoids = [centroid for centroid in centroids]
            try:
                temp_medoids.remove(medoid)
            except Exception as e:
                print("attempted to remove medoid from temp_medoids")
                print("medoid: ", medoid)
                print("temp_medoids: ", temp_medoids)
                print("centroids: ", centroids)
                raise e
            temp_medoids.append(tuple(image[point[0], point[1]]))
            temp_centroid_map = partition(image, temp_medoids, norm_function)
            # calculate the new cost function
            temp_costs = get_cost_per_medoid(image, temp_centroid_map, norm_function)

            # if the new cost function is less than the previous cost function
            if sum(temp_costs.values()) < sum(costs.values()):
                # set the new medoid
                centroids = temp_medoids
                centroid_map = temp_centroid_map
                costs = temp_costs

    return centroids, centroid_map


def should_stop(
    current_iteration: int,
    max_iterations: int,
    current_cost: float,
    previous_cost: float,
    threshold: float,
) -> bool:
    """
    Check if the algorithm should stop

    param: current_iteration: int
    param: max_iterations: int
    param: current_cost: float
    param: previous_cost: float
    param: threshold: float

    returns: bool
    """

    if current_iteration >= max_iterations:
        return True

    if current_cost < previous_cost:
        return True

    if abs(current_cost - previous_cost) < threshold:
        return True

    return False


def update_centroids(
    image: np.ndarray,
    centroid_map: dict[tuple, list[tuple]],
    norm_function=lambda x: p_norm(x, 2),
) -> dict[tuple, tuple]:
    """
    Update the centroids based on the current clustering

    param: image: np.ndarray of shape (m, n, 3)

    param: centroid_map: dict of centroids to list of points, each point is a coordinate in the image

    returns: dict of old centroids to new centroids
    """

    ans = {}

    for centroid, points in centroid_map.items():
        sum_points = np.array([0, 0, 0])
        for point in points:
            sum_points += image[point[0], point[1]]

        new_centroid = tuple(sum_points / len(points))
        ans[centroid] = new_centroid

    # # choose 100 random points of each centroid and set the new centroid to the closest point

    # for centroid, points in centroid_map.items():
    #     new_centroid_np = np.array(ans[centroid])
    #     for i in range(100):
    #         point = points[np.random.randint(0, len(points))]
    #         min_distance = float('inf')
    #         closest_point = None
    #         for point in points:
    #             distance = norm_function(new_centroid_np - image[point[0], point[1]])
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 closest_point = point

    #         ans[centroid] = tuple(image[closest_point[0], closest_point[1]])

    return ans


def partition(
    image: np.ndarray, centroids: list[tuple], norm_function
) -> dict[tuple, list[tuple]]:
    """
    Partition the image into k clusters based on the centroids

    param: image: np.ndarray of shape (m, n, 3)
    param: centroids: np.ndarray of shape (k, 3)
    param: norm_function: function to calculate the norm between two points
    """

    print("Partitioning the image")

    centroid_map = {centroid: [] for centroid in centroids}

    # this is for faster computation
    centroid_np_arraies = {centroid: np.array(centroid) for centroid in centroids}

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            point = image[i, j]
            min_distance = float("inf")
            closest_centroid = None

            for centroid in centroids:
                distance = norm_function(centroid_np_arraies[centroid] - point)
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = centroid

            centroid_map[closest_centroid].append((i, j))

            processed_pixels = i * image.shape[1] + j
            total_pixels = image.shape[0] * image.shape[1]
            # if processed_pixels % 10000 == 0:
            #     print("Processed pixels: ", processed_pixels, "Total pixels: ", total_pixels)

    return centroid_map


def select_centroids(image: np.ndarray, k: int) -> list[tuple]:
    """
    Select k random centroids from the image.

    """

    print("Selecting centroids")

    m, n, _ = image.shape
    centroids = []

    for i in range(k):
        x = np.random.randint(0, m)
        y = np.random.randint(0, n)
        while tuple(image[x, y]) in centroids:
            x = np.random.randint(0, m)
            y = np.random.randint(0, n)

        centroids.append(tuple(image[x, y]))

    return centroids


def get_cost_per_medoid(
    image: np.ndarray, centroid_map: dict[tuple, list[tuple]], norm_function
) -> dict[tuple, float]:
    """
    Calculate the cost of the current clustering

    param: image: np.ndarray of shape (m, n, 3)
    param: centroid_map: dict of centroids to list of points, each point is a coordinate in the image
    param: norm_function: function to calculate the norm
    """

    costs = {centroid: 0 for centroid in centroid_map.keys()}

    for centroid, points in centroid_map.items():
        centroid_np = np.array(centroid)
        for point in points:
            costs[centroid] += norm_function(centroid_np - image[point[0], point[1]])

    return costs


# def vector_difference(v1, v2) -> np.ndarray:
#     """
#     Sometimes we need to calculate the difference between two vectors,
#     those two vectors can be of any type, not necessarily numpy arrays,
#     so overloading the - operator won't work.
#     """
#     diff = [v1[i] - v2[i] for i in range(len(v1))]
#     return np.array(diff)


def p_norm(vector: np.array, p: int) -> float:
    """
    Calculate the p-norm of a vector

    param: vector: np.array
    param: p: int

    returns: float
    """

    return np.linalg.norm(vector, p)


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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("opened image of shape: ", image_rgb.shape, "image type: ", type(image_rgb))
    return image_rgb


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


def main():
    image = open_image_from_path(image_path)
    centroids, cluster_map = kmedoids(image, 7, lambda x: p_norm(x, 2))
    new_image = convert_cluster_map_to_image(image, cluster_map)
    resized = resize_image(new_image, 500)
    # save resized image as a new image
    cv2.imwrite("images/resized_image.png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    print("Running main")
    main()
