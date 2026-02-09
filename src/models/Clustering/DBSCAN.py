import os
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from typing import Dict, Set, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle

import matplotlib.pyplot as plt

# -------------------- Thread/Env Setup --------------------
NUM_THREADS = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(NUM_THREADS)

# -------------------- Worker functions --------------------
# _GLOBAL_G = None

# def _init_worker(G):
#     global _GLOBAL_G
#     _GLOBAL_G = G

# def _sssp_worker(node):
#     """Single source shortest path for a node."""
#     lengths = nx.single_source_shortest_path_length(_GLOBAL_G, node)
#     return node, lengths


def compute_all_pairs_shortest_paths_parallel(
    G: nx.Graph,
    n_jobs: int = None,
    return_matrix: bool = True,
    verbose: bool = True,
    config: Any = None
):
    """Parallel all-pairs shortest path length computation."""
    # global _GLOBAL_G # for ProcessPoolExecutor

    n_jobs = n_jobs or (os.cpu_count() or 1)
    nodes = list(G.nodes())

    if verbose:
        print(f"[INFO] Starting parallel APSP computation: {len(nodes)} nodes, {n_jobs} cores")

    # _GLOBAL_G = None
    # with ProcessPoolExecutor(
    #     max_workers=n_jobs,
    #     initializer=_init_worker,
    #     initargs=(G,)
    # ) as ex:
    #     if verbose:
    #         print("[INFO] Dispatching shortest path tasks...")
    #     results = list(ex.map(_sssp_worker, nodes, chunksize=max(1, len(nodes)//(n_jobs*8) or 1)))
    
    def _worker(node):
        return node, nx.single_source_shortest_path_length(G, node)
    
    results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for node, lengths in ex.map(_worker, nodes, chunksize=max(1, len(nodes)//n_jobs or 1)):
            results.append((node, lengths))

    if verbose:
        print("[INFO] Merging results...")

    apsp_dict = {src: dist for src, dist in results}

    if not return_matrix:
        return apsp_dict

    nodelist = nodes
    node_to_idx = {n: i for i, n in enumerate(nodelist)}
    N = len(nodelist)
    dist_matrix = np.full((N, N), np.inf, dtype=float)

    for src, dist_map in apsp_dict.items():
        i = node_to_idx[src]
        for tgt, d in dist_map.items():
            j = node_to_idx[tgt]
            dist_matrix[i, j] = float(d)

    # Replace infinities with large finite values
    inf_mask = np.isinf(dist_matrix)
    if np.any(inf_mask):
        max_finite = np.nanmax(dist_matrix[~inf_mask]) if np.any(~inf_mask) else 1.0
        dist_matrix[inf_mask] = max_finite * 2.0

    if verbose:
        print("[INFO] Done. Returning distance matrix.")
    SAVEPATH = os.path.join(config.savepath, 'LCC_sp_dist_matrix.npy')
    SAVEIMG = os.path.join(config.savepath, 'LCC_sp_dist_matrix.png')
    np.save(SAVEPATH, dist_matrix)
    
    # Plot histogram of distances
    upper_values = dist_matrix[np.triu_indices_from(dist_matrix)]
    fig = plt.figure(figsize=(12, 8))
    plt.hist(upper_values, bins=100)
    plt.title('All-pairs shortest path distance matrix histogram')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.savefig(SAVEIMG)
    plt.close(fig)
    
    return dist_matrix

# -------------------- DBSCAN using shortest paths --------------------
def dbscan_clustering_shortestpath(
    graph: nx.Graph,
    eps: float = 2.5,
    min_samples: int = 3,
    savePATH: str = '',
    config: Any = None
) -> Tuple[Dict[int, Set[Any]], nx.Graph]:
    """
    Performs DBSCAN clustering on a NetworkX graph using all-pairs shortest path distances.

    :param graph: NetworkX graph object.
    :param eps: DBSCAN neighborhood radius.
    :param min_samples: Minimum number of samples to form a cluster.
    :param savePATH: Optional path to save the cluster dictionary as a pickle file.
    :return: (clusters_dict, updated_graph_with_labels)
    """

    G = graph.copy()

    # --- Compute all-pairs shortest path distances ---
    
    pre_calc_matrix = os.path.join(config.savepath, 'LCC_sp_dist_matrix.npy')
        
    if os.path.exists(pre_calc_matrix):
        print(f"[INFO] Loading precomputed shortest-path distance matrix from {pre_calc_matrix}...")
        dist_matrix = np.load(pre_calc_matrix)
        
    else:    
        dist_matrix = compute_all_pairs_shortest_paths_parallel(G,
                                                                n_jobs=NUM_THREADS,
                                                                return_matrix=False,
                                                                verbose=True,
                                                                config=config
                                                                )
    nodes = list(G.nodes())
    
    # --- Run DBSCAN ---
    print("[INFO] Running DBSCAN on precomputed shortest-path distances...")
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed',
        n_jobs=NUM_THREADS,
    )
    labels = dbscan.fit_predict(dist_matrix)
    core_sample = dbscan.components_

    # --- Map cluster labels back to nodes ---
    clusters: Dict[int, Set[Any]] = {}
    
    for node, label in zip(nodes, labels):
        G.nodes[node]['DBSCAN'] = int(label)
        clusters.setdefault(int(label), set()).add(node)

    save_file_dict = {'clusters': clusters,
                      'params': {'eps': eps, 'min_samples': min_samples},
                      'core_samples': core_sample,
                     }
    
    # --- Save clusters if path is given ---
    if savePATH:
        save_file = os.path.join(savePATH, 'DBSCAN_clusters.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(save_file_dict, f)
        print(f"[INFO] Saved clusters to {save_file}")

    print(f"[INFO] Clustering complete. Found {len(set(labels) - {-1})} clusters and {sum(labels==-1)} noise points.")
    
    return clusters, G



def hdbscan_clustering_shortestpath(
    graph: nx.Graph,
    min_cluster_size: int = 5,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    allow_single_cluster: bool = False,
    savePATH: str = '',
    config: Any = None
) -> Tuple[Dict[int, Set[Any]], nx.Graph]:
    """
    Performs HDBSCAN clustering on a NetworkX graph using all-pairs shortest path distances.

    :param graph: NetworkX graph object.
    :param min_cluster_size: Minimum size of clusters.
    :param min_samples: Controls sensitivity to noise (if None, defaults to min_cluster_size).
    :param cluster_selection_epsilon: Threshold for flat cluster extraction.
    :param cluster_selection_method: 'eom' or 'leaf'.
    :param savePATH: Optional path to save the cluster dictionary as a pickle file.
    :param config: Config object containing savepath for precomputed matrix.
    :return: (clusters_dict, updated_graph_with_labels)
    """

    G = graph.copy()

    # --- Load or compute shortest-path distance matrix ---
    pre_calc_matrix = None
    if config is not None and hasattr(config, "savepath"):
        pre_calc_matrix = os.path.join(config.savepath, 'LCC_sp_dist_matrix.npy')

    if pre_calc_matrix and os.path.exists(pre_calc_matrix):
        print(f"[INFO] Loading precomputed shortest-path distance matrix from {pre_calc_matrix}...")
        dist_matrix = np.load(pre_calc_matrix)
    else:
        dist_matrix = compute_all_pairs_shortest_paths_parallel(
            G,
            n_jobs=NUM_THREADS,
            return_matrix=True,
            verbose=True,
            config=config
        )

    nodes = list(G.nodes())

    # --- Run HDBSCAN ---
    print("[INFO] Running HDBSCAN on precomputed shortest-path distances...")
    clusterer = HDBSCAN(
        metric='precomputed',
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        n_jobs=NUM_THREADS,
        # store_centers='medoid'
    )

    clusterer.fit(dist_matrix)

    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    # --- Map cluster labels back to nodes ---
    clusters: Dict[int, Set[Any]] = {}
    for i, (node, label) in enumerate(zip(nodes, labels)):
        G.nodes[node]['HDBSCAN'] = int(label)
        
        if probabilities[i] is np.nan:
            probabilities[i] = -1
        G.nodes[node]['probability'] = float(probabilities[i])
        
        clusters.setdefault(int(label), set()).add(node)

    save_file_dict = {
        'clusters': clusters,
        'params': {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'cluster_selection_epsilon': cluster_selection_epsilon,
            'cluster_selection_method': cluster_selection_method
        },
        'probabilities': probabilities,
        # 'core_samples': clusterer.medoids_
    }

    # --- Save clusters if path is given ---
    if savePATH:
        save_file = os.path.join(savePATH, 'HDBSCAN_clusters.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(save_file_dict, f)
        print(f"[INFO] Saved HDBSCAN results to {save_file}")

    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    print(f"[INFO] HDBSCAN complete. Found {n_clusters} clusters and {n_noise} noise points.")

    return clusters, G