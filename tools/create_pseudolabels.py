import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import argparse

import pandas as pd
import kaldiio

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score

from cuml.cluster import KMeans

import torch


def main(args: argparse.Namespace):
    """
    Create pseudo labels from embeddings.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    # Load embeddings
    input_ext = args.embeddings_file.split('.')[-1]
    if input_ext == "pt":
        embeddings = torch.load(args.embeddings_file, map_location="cpu")
    elif input_ext == "scp":
        embeddings = {}
        for key, numpy_array in kaldiio.load_scp_sequential(args.embeddings_file):
            embeddings[key] = torch.tensor(numpy_array).unsqueeze(0)

    dataset_size = len(embeddings)
    files = list(embeddings.keys())
    labels = [file.split('/')[-3] for file in files]

    embeddings = torch.cat(list(embeddings.values()))
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.numpy()
    print(f"Embeddings shape: {embeddings.shape}")

    # K-means
    if not args.silent:
        print("Running K-Means...")
    kmeans_start_time = time.time()
    kmeans = KMeans(
        n_clusters=args.nb_clusters,
        random_state=0,
        max_samples_per_batch=1000000,
    ).fit(embeddings)
    assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print(f"K-Means duration: {(time.time() - kmeans_start_time)/60:.2f} min")

    # AHC
    if args.nb_clusters_ahc > 0:
        if not args.silent:
            print("Running AHC...")
        ahc_start_time = time.time()
        assignments_ahc = AgglomerativeClustering(
            n_clusters=args.nb_clusters_ahc
        ).fit_predict(centroids)
        assignments = [assignments_ahc[a] for a in assignments]
        print(f"AHC duration: {(time.time() - ahc_start_time)/60:.2f} min")

    # NMI
    nmi_score = normalized_mutual_info_score(labels, assignments)
    print(f"NMI: {nmi_score}")

    # Save pseudo labels to file
    if args.output_format == "csv":
        files = [f.replace('data/', '') for f in files]
        df = pd.DataFrame({ "File": files, "Speaker": assignments })
        df.to_csv(args.output_file, index=False)
    elif args.output_format == "kaldi":
        files = ['/'.join(f.split('/')[-3:]) for f in files]
        with open(args.output_file, 'w') as f:
            for file, pseudo_label in zip(files, assignments):
                f.write(f"{file} {pseudo_label}\n")
    else:
        raise Exception(f'Output format {args.output_format} not supported')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "embeddings_file",
        type=str,
        help="Path to embeddings file (.pt).",
    )
    parser.add_argument(
        "output_file",
        help="Path to output file.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="csv",
        help="Output file format",
    )
    parser.add_argument(
        "--nb_clusters",
        type=int,
        default=50000,
        help="Number of clusters for k-means.",
    )
    parser.add_argument(
        "--nb_clusters_ahc",
        type=int,
        default=7500,
        help="Number of clusters for AHC.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Whether to hide status messages and progress bars.",
    )
    args = parser.parse_args()

    main(args)
