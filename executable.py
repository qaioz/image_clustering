
import argparse
from src.clustering.main import ClusteringAlgorithm, run_clustering_and_generate_image
from src.compression.main import Compression_Algorithm, run_compress_image, run_decompress_image


def main():
    parser = argparse.ArgumentParser(description="Image Clustering and Compression Tool")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Clustering command
    cluster_parser = subparsers.add_parser("cluster", help="Cluster an image using k-means or k-medoids")
    cluster_parser.add_argument("image_path", type=str, help="Path to the input image")
    cluster_parser.add_argument("--algorithm", type=str, choices=['kmeans','kmedoids'], default='kmeans', help="Clustering algorithm (default: kmeans)")
    cluster_parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters (default: 10)")
    cluster_parser.add_argument("--iterations", type=int, default=10, help="Max iterations (default: 10)")
    cluster_parser.add_argument("--norm", type=float, default=2, help="Norm (default: 2)")
    cluster_parser.add_argument("--output_file", type=str, help="Path to save the output clustered image")

    # Compression command
    compress_parser = subparsers.add_parser("compress", help="Compress an image using k-means clustering")
    compress_parser.add_argument("image_path", type=str, help="Path to the input image")
    compress_parser.add_argument("--algorithm", type=str, choices=['kmeans','kmedoids'], default='kmeans', help="Compression algorithm (default: kmeans)")
    compress_parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters (default: 10)")
    compress_parser.add_argument("--iterations", type=int, default=10, help="Max iterations (default: 10)")
    compress_parser.add_argument("--norm", type=float, default=2, help="Norm (default: 2)")
    compress_parser.add_argument("--output_file", type=str, help="Path to save the compressed file")

    # Decompression command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a compressed image file")
    decompress_parser.add_argument("compressed_file_path", type=str, help="Path to the compressed file")

    args = parser.parse_args()

    if args.command == "cluster":
        run_clustering_and_generate_image(
            image_path=args.image_path,
            clustering_algorithm=ClusteringAlgorithm.KMEANS if args.algorithm == 'kmeans' else ClusteringAlgorithm.KMEDOIDS,
            n_clusters=args.n_clusters,
            max_iterations=args.iterations,
            norm=args.norm,
            output_file=args.output_file
        )
    elif args.command == "compress":
        run_compress_image(
            image_path=args.image_path,
            algorithm=ClusteringAlgorithm.KMEANS if args.algorithm == 'kmeans' else ClusteringAlgorithm.KMEDOIDS,
            n_of_clusters=args.n_clusters,
            iterations=args.iterations,
            norm=args.norm,
            output_file=args.output_file
        )
    elif args.command == "decompress":
        run_decompress_image(args.compressed_file_path)

if __name__ == "__main__":
    main()