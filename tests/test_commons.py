import pytest
import numpy as np
from src.commons import (
    get_image_unique_colors_and_frequencies,
    select_clusters,
    cost_function,
    partition,
    get_new_image_from_original_image_and_clusters,
    generate_new_clusters,
    get_color_clusters,
)

# test the cost function using centroids = [[1,1,1],[0,100,100]]

# norm = 2
# color_centroids_0 = [[1,1,1],[1,1,1]],         color_frequencies_0 = 1 => cost = 0
# color_centroids_1 = [[1,4,5],[1,1,1]],         color_frequencies_1 = 2 => cost = 10
# color_centroids_2 = [[2,3,3],[1,1,1]],         color_frequencies_2 = 3 => cost = 9
# color_centroids_3 = [[5,112,100],[0,100,100]], color_frequencies_3 = 4 => cost = 13 * 4 = 52
# color_centroids_4 = [[0,100,100],[0,100,100]], color_frequencies_4 = 5 => cost = 0

# expected_cost = 0 + 10 + 9 + 52 + 0 = 71
# the above is old implementation, the new implementation is different,

# test the cost function using centroids = [[1,1,1],[0,100,100]]

# colors = [[1,1,1],[1,4,5],[2,3,3],[5,112,100],[0,100,100]]
# color_frequencies = [1,2,3,4,5]
# clusters = [[1,1,1],[0,100,100]]
# color_cluter_indices = [0,0,0,1,1]
# norm = 2
# expected_cost = 0 + 10 + 9 + 52 + 0 = 71


@pytest.mark.parametrize(
    "colors, clusters, color_cluter_indices, color_frequencies, norm, expected_cost",
    [
        (
            np.array(
                [
                    [1, 1, 1],
                    [1, 4, 5],
                    [2, 3, 3],
                    [5, 112, 100],
                    [0, 100, 100],
                ]
            ),
            np.array(
                [
                    [1, 1, 1],
                    [0, 100, 100],
                ]
            ),
            np.array([0, 0, 0, 1, 1]),
            np.array([1, 2, 3, 4, 5]),
            2.0,
            71,
        )
    ],
)
def test_cost_function(
    colors, clusters, color_cluter_indices, color_frequencies, norm, expected_cost
):
    assert (
        cost_function(colors, clusters, color_cluter_indices, color_frequencies, norm)
        == expected_cost
    )


# test partition

# unique_colors = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[90,90,90],[100,100,100],[110,110,110]]
# centroids = [[1,1,1],[100,100,100]]
# norm = 2

# expected_color_centroid_indices [0,0,0,0,1,1,1]


@pytest.mark.parametrize(
    "unique_colors, centroids, norm, expected_color_centroid_indices",
    [
        (
            np.array(
                [
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [90, 90, 90],
                    [100, 100, 100],
                    [110, 110, 110],
                ]
            ),
            np.array(
                [
                    [1, 1, 1],
                    [100, 100, 100],
                ]
            ),
            2.0,
            np.array([0, 0, 0, 0, 1, 1, 1]),
        )
    ],
)
def test_partition(unique_colors, centroids, norm, expected_color_centroid_indices):
    assert np.array_equal(
        partition(unique_colors, centroids, norm), expected_color_centroid_indices
    )


# test get_image_unique_colors_and_frequencies

# image_0 = [[1,1,1],[10,10,10],[10,10,10]]
# image_1 = [[10,10,10],[10,10,10],[15,15,15]]
# image_2 = [[100,100,100],[100,100,100],[100,100,100]]

# expected_unique_colors = [[1,1,1],[10,10,10],[15,15,15],[100,100,100]]
# expected_color_frequencies = [1,4,1,3]


@pytest.mark.parametrize(
    "image, expected_unique_colors, expected_color_frequencies",
    [
        (
            np.array(
                [
                    [[1, 1, 1], [10, 10, 10], [10, 10, 10]],
                    [[10, 10, 10], [10, 10, 10], [15, 15, 15]],
                    [[100, 100, 100], [100, 100, 100], [100, 100, 100]],
                ]
            ),
            np.array([[1, 1, 1], [10, 10, 10], [15, 15, 15], [100, 100, 100]]),
            np.array([1, 4, 1, 3]),
        )
    ],
)
def test_get_image_unique_colors_and_frequencies(
    image, expected_unique_colors, expected_color_frequencies
):
    unique_colors, color_frequencies = get_image_unique_colors_and_frequencies(image)

    assert np.array_equal(unique_colors, expected_unique_colors)
    assert np.array_equal(color_frequencies, expected_color_frequencies)


# test get_new_image_from_original_image_and_medoids

# image_0 = [[1,1,1],[10,10,10],[10,10,10]]
# image_1 = [[10,10,10],[10,10,10],[15,15,15]]
# image_2 = [[100,100,100],[100,100,100],[100,100,100]]

# medoids = [[1,1,1],[9,9,9],[90,110,90]]

# expected_new_image_0 = [[1,1,1],[9,9,9],[9,9,9]]
# expected_new_image_1 = [[9,9,9],[9,9,9],[9,9,9]]
# expected_new_image_2 = [[90,110,90],[90,110,90],[90,110,90]]


@pytest.mark.parametrize(
    "image, medoids, expected_new_image",
    [
        (
            np.array(
                [
                    [[1, 1, 1], [10, 10, 10], [10, 10, 10]],
                    [[10, 10, 10], [10, 10, 10], [15, 15, 15]],
                    [[100, 100, 100], [100, 100, 100], [100, 100, 100]],
                ]
            ),
            np.array([[1, 1, 1], [9, 9, 9], [90, 110, 90]]),
            np.array(
                [
                    [[1, 1, 1], [9, 9, 9], [9, 9, 9]],
                    [[9, 9, 9], [9, 9, 9], [9, 9, 9]],
                    [[90, 110, 90], [90, 110, 90], [90, 110, 90]],
                ]
            ),
        )
    ],
)
def test_get_new_image_from_original_image_and_medoids(
    image, medoids, expected_new_image
):
    assert np.array_equal(
        get_new_image_from_original_image_and_clusters(image, medoids, 2),
        expected_new_image,
    )


# test generate_new_clusters

# centroids = [[1,1,1],[0,100,100]]
# unique_colors = [[1,1,1],[2,2,2],[3,3,3],[100,100,100],[200,200,200]]
# color_frequencies = [1,2,1,2,2]
# color_centroid_indices = [0,0,0,1,1]
# norm = 2

# expected_new_clusters_0 = [1,1,1] * 1 + [2,2,2] * 2 + [3,3,3] * 1 = [8,8,8] / 4 = [2,2,2]
# expected_new_clusters_1 = [100,100,100] * 2 + [200,200,200] * 2 = [600,600,600] / 4 = [150,150,150]


@pytest.mark.parametrize(
    "unique_colors, color_frequencies, color_centroid_indices, centroids, expected_new_clusters",
    [
        (
            np.array(
                [
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [100, 100, 100],
                    [200, 200, 200],
                ]
            ),
            np.array([1, 2, 1, 2, 2]),
            np.array([0, 0, 0, 1, 1]),
            np.array([[1, 1, 1], [0, 100, 100]]),
            np.array([[2, 2, 2], [150, 150, 150]]),
        )
    ],
)
def test_generate_new_clusters(
    unique_colors,
    color_frequencies,
    color_centroid_indices,
    centroids,
    expected_new_clusters,
):
    assert np.array_equal(
        generate_new_clusters(
            unique_colors, color_frequencies, centroids, color_centroid_indices
        ),
        expected_new_clusters,
    )


# test get_color_centroids

# colors = [[1,1,1],[2,2,2],[3,3,3],4,4,4],[90,90,90],[100,100,100],[110,110,110]]
# centroids = [[1,1,1],[100,100,100]]
# color_centroid_indices = [0,0,0,0,1,1,1]
# expected_color_centroids = [ [[1,1,1],[1,1,1]], [[2,2,2],[1,1,1]], [[3,3,3],[1,1,1]], [[4,4,4],[1,1,1]], [[90,90,90],[100,100,100]], [[100,100,100],[100,100,100]], [[110,110,110],[100,100,100]] ]
@pytest.mark.parametrize(
    "colors, centroids, color_centroid_indices, expected_color_centroids",
    [
        (
            np.array(
                [
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [90, 90, 90],
                    [100, 100, 100],
                    [110, 110, 110],
                ]
            ),
            np.array(
                [
                    [1, 1, 1],
                    [100, 100, 100],
                ]
            ),
            np.array([0, 0, 0, 0, 1, 1, 1]),
            np.array(
                [
                    [[1, 1, 1], [1, 1, 1]],
                    [[2, 2, 2], [1, 1, 1]],
                    [[3, 3, 3], [1, 1, 1]],
                    [[4, 4, 4], [1, 1, 1]],
                    [[90, 90, 90], [100, 100, 100]],
                    [[100, 100, 100], [100, 100, 100]],
                    [[110, 110, 110], [100, 100, 100]],
                ]
            ),
        )
    ],
)
def test_get_color_centroids(
    colors, centroids, color_centroid_indices, expected_color_centroids
):
    assert np.array_equal(
        get_color_clusters(colors, centroids, color_centroid_indices),
        expected_color_centroids,
    )
