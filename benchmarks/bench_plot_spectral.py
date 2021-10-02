"""Benchmarks for spectral clustering methods."""

import numpy as np
from scipy import sparse
from memory_profiler import memory_usage
import timeit

import csv
import gc

from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import kernel_metrics, rbf_kernel
from sklearn.datasets import make_blobs


def to_numpy(X_file, y_file):
    i_list, j_list = [], []
    for row in X_file:
        # Convert from one-based to zero-based indexing.
        i = int(row[0]) - 1
        j = int(row[1]) - 1
        i_list.append(i)
        j_list.append(j)
        # symmetrize
        j_list.append(i)
        i_list.append(j)

    vals = [1 for _ in i_list]
    N = max(i_list) + 1

    S = sparse.coo_matrix((vals, (i_list, j_list)),
                            shape=(N, N), dtype=np.float32)
    y = []
    for row in y_file:
        y.append(int(row[1]) - 1)
    
    return S, np.array(y)


def get_data():
    #for size in (50, 100, 500, 1000):#, 5000):
    #    X_file = csv.reader(open(f"simulated_blockmodel_graph_{size}_nodes.tsv"),
    #                        delimiter="\t")
    #    y_file = csv.reader(open(f"simulated_blockmodel_graph_{size}_nodes_truePartition.tsv"),
    #                        delimiter="\t")
    #   
    #    yield to_numpy(X_file, y_file)

    cases = [
        (1000, "static_highOverlap_lowBlockSizeVar"),
        (1000, "static_highOverlap_highBlockSizeVar"),
        (5000, "static_highOverlap_lowBlockSizeVar"),
    ]

    for size, name in cases:
        X_file = csv.reader(open(f"{name}_{size}_nodes.tsv"),
                            delimiter="\t")
        y_file = csv.reader(open(f"{name}_{size}_nodes_truePartition.tsv"),
                            delimiter="\t")
        yield to_numpy(X_file, y_file), name


def profile_and_score(S, y, assign_labels="cluster_qr", n_clusters=2):

    def cluster():
        return SpectralClustering(
                random_state=0,
                n_clusters=n_clusters,
                affinity="precomputed",
                assign_labels=assign_labels,
            ).fit(S)

    gc.collect()
    #print("Starting profiling...")
    #print("Timing...")
    time = np.mean(timeit.repeat(cluster, repeat=3, number=1))
    #print("Profiling memory usage...")
    memory = np.max(memory_usage(cluster))
    #print("Scoring accuracy...")
    score = adjusted_rand_score(y, cluster().labels_)
    #print("Profiling complete")
    return time, memory, score


def run_benchmark():
    #X, y = make_blobs(
    #    n_samples=1000, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    #)
    for (S, y), name in get_data():
        n_clusters = np.max(y) + 1
        print(f"Test {name} of size {S.shape} with {n_clusters} clusters")

        for method in ("kmeans", "discretize", "cluster_qr"):
            time, memory, score = profile_and_score(S, y, method, n_clusters=n_clusters)
            print(f"{method:10}: {score:.3f} ({time:.3f} sec., {memory:.3f} MB)")
        
        print('\n')


if __name__ == "__main__":
    run_benchmark()
