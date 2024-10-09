import pytest
import numpy as np
import src.kmedoids as kmedoids


# test the cost function using centroids = [[1,1,1],[0,100,100]]

# norm = 2
# color_centroids_0 = [[1,1,1],[1,1,1]], color_frequencies_0 = 1 => cost = 0
# color_centroids_1 = [[1,4,5],[1,1,1]], color_frequencies_1 = 2 => cost = 10
# color_centroids_2 = [[2,3,3],[1,1,1]], color_frequencies_2 = 3 => cost = 9
# color_centroids_3 = [[5,112,100],[0,100,100]], color_frequencies_3 = 4 => cost = 13 * 4 = 52
# color_centroids_4 = [[0,100,100],[0,100,100]], color_frequencies_4 = 5 => cost = 0

# expected_cost = 0 + 10 + 9 + 52 + 0 = 71


@pytest.mark.parametrize(
    "color_centroids, color_frequencies, norm, expected_cost",
    [
        (
            np.array(
                [
                    [[1, 1, 1], [1, 1, 1]],
                    [[1, 4, 5], [1, 1, 1]],
                    [[2, 3, 3], [1, 1, 1]],
                    [[5, 112, 100], [0, 100, 100]],
                    [[0, 100, 100], [0, 100, 100]],
                ]
            ),
            np.array([1, 2, 3, 4, 5]),
            2.0,
            71,
        )
    ],
)
def test_cost_function(color_centroids, color_frequencies, norm, expected_cost):
    assert (
        kmedoids.cost_function(color_centroids, color_frequencies, norm)
        == expected_cost
    )


# test partition

# unique_colors = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[90,90,90],[100,100,100],[110,110,110]]
# centroids = [[1,1,1],[100,100,100]]
# norm = 2

# expected_color_centroids = [[[1,1,1],[1,1,1]],[[2,2,2],[1,1,1]],[[3,3,3],[1,1,1]],[[4,4,4],[1,1,1]],[[90,90,90],[100,100,100]],[[100,100,100],[100,100,100]],[[110,110,110],[100,100,100]]]


@pytest.mark.parametrize(
    "unique_colors, centroids, norm, expected_color_centroids",
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
def test_partition(unique_colors, centroids, norm, expected_color_centroids):
    assert np.array_equal(
        kmedoids.partition(unique_colors, centroids, norm), expected_color_centroids
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
    unique_colors, color_frequencies = kmedoids.get_image_unique_colors_and_frequencies(
        image
    )

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
        kmedoids.get_new_image_from_original_image_and_medoids(image, medoids),
        expected_new_image,
    )
    

