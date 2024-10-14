import pytest
import numpy as np
from src.compression.image_compression import (
    compress_clustered_image,
    compress_image
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


# test the compress_image function which has to generate a file, so use temorary directory

# input image is mock, its .shape attribute is (3,3,3). The algorithm is also mocked and will return the following clusters and image

# using the mocked algorithm, that will return the following image and clusters
# clusters,image
# clusters = [[1,1,1],[2,2,2]]
# image_0 = [[2,2,2],[1,1,1],[2,2,2]]
# image_1 = [[1,1,1],[1,1,1],[2,2,2]]
# image_2 = [[2,2,2],[2,2,2],[1,1,1]]


#then the funtion should generate a file with the following content:

# expected file content:
#3,3
#010101,020202
#1:1,0:1,1:1,0:2,1:3,0:1


def test_compress_image(tmpdir):
    image = Mock()
    image.shape = (3,3,3)
    algorithm = Mock(return_value=(np.array([[1,1,1],[2,2,2]]), np.array([[[2,2,2],[1,1,1],[2,2,2]],[[1,1,1],[1,1,1],[2,2,2]],[[2,2,2],[2,2,2],[1,1,1]]])))
    output_file = tmpdir.join("output.txt") 
    # mock open_image_from_path to return the mock image
    with patch("src.compression.image_compression.open_image_from_path", return_value=image):
        compress_image("path", algorithm=algorithm, output_file=str(output_file))
        
    assert output_file.read() == "3,3,2\n010101,020202\n1:1,0:1,1:1,0:2,1:3,0:1\n"
    