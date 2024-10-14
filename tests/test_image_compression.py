import pytest
import numpy as np
from src.compression.image_compression import (
    compress_clustered_image,
    _save_compressed_image_binary,
)
from src.clustering.image_clustering import kmeans, kmedoids
from unittest.mock import Mock, patch


# test the compress_clustered_image function

# image_0 = [[1,1,1],[10,10,10],[10,10,10]]
# image_1 = [[10,10,10],[10,10,10],[15,15,15]]
# image_2 = [[100,100,100],[100,100,100],[100,100,100]]

# clusters = [[1,1,1],[10,10,10],[15,15,15],[100,100,100]]

# expected_compressed_0 = [[0,1],[1,4],[2,1],[3,3]]


@pytest.mark.parametrize(
    "image, clusters, expected_compressed",
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
            np.array([[0, 1], [1, 4], [2, 1], [3, 3]]),
        )
    ],
)
def test_compress_clustered_image(image, clusters, expected_compressed):
    compressed = compress_clustered_image(image, clusters)

    assert np.array_equal(compressed, expected_compressed)


# test the _save_compressed_image_binary function

# input compressed_image is a 2D numpy array with shape (3,2)

# original_dimensions: (3,3)
# clusters: [[1,1,1],[2,2,2]]
# compressed_image: [[1,5],[0,1],[1,1],[0,2]]

# expected file content byte by byte in decimal values:
# 2 bytes for image height, 2 bytes for image width, 1 byte for number of clusters
# 0 3 0 3 2

# then for each cluster 3 bytes for the color values
# 1 1 1 2 2 2

# then for each row in the compressed image 1 byte for the cluster index and 3 bytes for the count
# 1 0 0 5
# 0 0 0 1
# 1 0 0 1
# 0 0 0 2


def test_save_compressed_image_binary(tmpdir):
    compressed_image = np.array([[1, 5], [0, 1], [1, 1], [0, 2]])
    original_dimensions = (3, 3)
    clusters = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8)
    output_file = tmpdir.join("output.bin")

    # Call the function to save the binary file
    _save_compressed_image_binary(
        compressed_image=compressed_image,
        original_dimensions=original_dimensions,
        clusters=clusters,
        output_file=str(output_file),
    )

    # Prepare the expected content as a list of byte values
    expected_content = [
        0,
        3,
        0,
        3,
        2,  # number of clusters
        1,
        1,
        1,  # cluster 1 color
        2,
        2,
        2,  # cluster 2 color
        1,
        0,
        0,
        5,  # compressed row 0
        0,
        0,
        0,
        1,  # compressed row 1
        1,
        0,
        0,
        1,  # compressed row 2
        0,
        0,
        0,
        2,  # compressed row 3
    ]

    # Open the file and read it byte by byte
    with open(output_file, "rb") as file:
        for expected_byte in expected_content:
            byte = file.read(1)  # Read one byte
            assert byte == expected_byte.to_bytes(
                1, byteorder="big"
            ), f"Expected {expected_byte} but got {byte[0]}"

        assert file.read() == b""
