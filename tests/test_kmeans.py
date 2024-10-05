import pytest
import numpy as np
import src.kmeans as kmeans


@pytest.mark.parametrize(
    "centroids, elements_per_centroid, norm, expected_cost",
    [
        (
            np.array([[1, 1, 1], [0, 10, 10]]),
            [
                np.array([[1, 4, 5], [2, 3, 3]]),
                np.array([[0, 10, 10], [1, 8, 8], [0, 10, 5]]),
            ],
            2,
            16,
        )
    ],
)
def test_cost_function(centroids, elements_per_centroid, norm, expected_cost):
    assert kmeans.cost_function(centroids, elements_per_centroid, norm) == expected_cost


# image_0 = [[1,1,1],[10,10,10],[10,10,10]]
# image_1 = [[10,10,10],[10,10,10],[15,15,15]]
# image_2 = [[100,100,100],[100,100,100],[100,100,100]]
# centroids = [[2,2,2],[9,9,9],[30,30,30]]
# norm = 2
# results:
# pixels_per_partition_0 = [[1,1,1]]
# pixels_per_partition_1 = [[10,10,10] 4 times, [15,15,15]]
# pixels_per_partition_2 = [[100,100,100] 3 times]
# point_clusters_0 = [0,1,1]
# point_clusters_1 = [1,1,1]
# point_clusters_2 = [2,2,2]
@pytest.mark.parametrize(
    "image, centroids, norm, exp_pixels_per_partition, exp_point_clusters",
    [
        (
            np.array(
                [
                    [[1, 1, 1], [10, 10, 10], [10, 10, 10]],
                    [[10, 10, 10], [10, 10, 10], [15, 15, 15]],
                    [[100, 100, 100], [100, 100, 100], [100, 100, 100]],
                ]
            ),
            np.array([[2, 2, 2], [9, 9, 9], [30, 30, 30]]),
            2,
            [
                np.array([[1, 1, 1]]),
                np.array(
                    [
                        [15, 15, 15],
                        [10, 10, 10],
                        [10, 10, 10],
                        [10, 10, 10],
                        [10, 10, 10],
                    ]
                ),
                np.array([[100, 100, 100], [100, 100, 100], [100, 100, 100]]),
            ],
            [np.array([0, 1, 1]), np.array([1, 1, 1]), np.array([2, 2, 2])],
        )
    ],
)
def test_partition(
    image, centroids, norm, exp_pixels_per_partition, exp_point_clusters
):
    pixel_per_partition, point_clusters = kmeans.partition(image, centroids, norm)

    for i in range(len(pixel_per_partition)):
        sorted_partition = np.sort(pixel_per_partition[i], axis=0)
        sorted_expected = np.sort(exp_pixels_per_partition[i], axis=0)

        assert np.array_equal(sorted_partition, sorted_expected)

    for i in range(len(point_clusters)):
        assert np.array_equal(point_clusters[i], exp_point_clusters[i])


# first test
# elements_per_centroid_0 = [[1,1,1],[3,3,3]]
# elements_per_centroid_1 = [[10,10,10],[12,12,12], [14,14,14]]
# expected_0 = [2,2,2]
# expected_1 = [11,11,11]

# second test for very large vectors of (255,255,255)
# elements_per_centroid_0 = generate a list of 100000 random 3d vectors from values [0,0,0] to [255,255,255]
# elements_per_centroid_1 = generate a list of 1000000 random 3d vectors from values [0,0,0] to [255,255,255]
# expected_0 = around 255/2 = [127.5, 127.5, 127.5] but not exactly because of the random nature of the vectors
# expected_1 = around 255/2 = [127.5, 127.5, 127.5] but not exactly because of the random nature of the vectors


@pytest.mark.parametrize(
    "elements_per_centroid, expected",
    [
        (
            [
                np.array([[1, 1, 1], [3, 3, 3]]),
                np.array([[10, 10, 10], [12, 12, 12], [14, 14, 14]]),
            ],
            [np.array([2, 2, 2]), np.array([12, 12, 12])],
        )
    ],
)
def test_generate_new_centroids(elements_per_centroid, expected):
    new_centroids = kmeans.generate_new_centroids(elements_per_centroid)
    assert np.array_equal(new_centroids, expected)


@pytest.mark.parametrize(
    "num_of_vectors_1, num_of_vectors_2, error_range",
    [(10 ** 6, 10 ** 7, 0.8), (10 ** 6, 10 ** 7, 0.7)],
)
def test_generate_new_centroids_probailistic_large_values(
    num_of_vectors_1, num_of_vectors_2, error_range
):
    """
    I did not trust if we np.mean if the sum of the values is too large, so I created this test to check if the function works
    And looks like it does

    I will generate a list of 10*6 random 3d vectors from values [0,0,0] to [255,255,255]
    I will generate a list of 10*7 random 3d vectors from values [0,0,0] to [255,255,255]

    Expected values are around 255/2 = [127.5, 127.5, 127.5]
    I will check if the new centroids are around this value with an error of some range
    """

    elements_per_centroid = [
        np.random.randint(0, 255, (num_of_vectors_1, 3)),
        np.random.randint(0, 255, (num_of_vectors_2, 3)),
    ]

    expected = [np.array([127.5, 127.5, 127.5]), np.array([127.5, 127.5, 127.5])]

    new_centroids = kmeans.generate_new_centroids(elements_per_centroid)

    assert np.allclose(new_centroids, expected, atol=error_range)
