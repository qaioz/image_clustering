# Image Clustering and Image Compression with K-means & K-medoids

This project demonstrates image clustering and and it's application in image compression using K-means and K-Medoids algorithms. The objective is to reduce the number of unique colors in image by grouping similar colors into clusters, and come up with a way to effectively reduce the space.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)

  ## Overview

This project offers 

1. fast and parameterized K-means and K-medoids clustering for images. 
2. allowing you to compress .bmp files while specifying the number of clusters and the algorithm to use. 

  - More clusters (e.g., K=20) result in higher-quality images, where the difference is nearly invisible to the human eye.
  - However, increasing the cluster count slows down the algorithm and reduces space savings.


The compression process reads a .bmp image, applies K-means clustering, and saves the clustered image as a binary file in a custom .gcp format. This format uses linear compression, optimizing for repeating colors to keep the file size small.

Project uses numpy vectorization, avoiding python looks as much as possible. This is 1000 times fasater than the initial version with python loops.

Note: K-medoids is trash and very slow for every real case. It is just for the demo purposes.


## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- `pip` for installing Python dependencies
- OpenCV (`cv2`) for image processing
- NumPy for numerical operations

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/image-clustering-compression.git
    cd image-clustering-compression
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running Compression and Decompression

To compress an image using K-means clustering and save it in binary format, you can run the following script:

```bash
python src/compress_and_decompress_image.py --image_path path/to/your/image.png --num_clusters 4

