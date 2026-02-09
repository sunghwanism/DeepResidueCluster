
SEED = 42

clustering_params = {
    'CDBN': {
        'k': None,  # If None, computes exact betweenness; otherwise, uses k random nodes as sources
        'normalized': True,
        'weight': None, # or name of edge attribute
        'seed': SEED
        },
    'DBSCAN': {
        'eps': 2,  # max distance between two samples for one to be considered as in the neighborhood of the other
        'min_samples': 4, # min_samples in cluster
        },
    'HDBSCAN': {
        'min_cluster_size': 5, # min cluster size
        'min_samples': 3, # min samples in cluster
        'cluster_selection_epsilon': 0.0, # epsilon value to control the distance threshold
        'cluster_selection_method': 'eom', # 'eom' or 'leaf'
        'max_cluster_size': None, # If not None, the largest cluster will be at most this size
        'allow_single_cluster': False
        },
    'MCL': {
        'expansion_power': 5,  # Expansion power (e)
        'inflation_power': 3,  # Inflation power (r)
        'pruning_threshold': 0.13,  # Pruning threshold (t)
        'convergence_threshold': 1e-6,  # Convergence threshold (c)
        'max_iterations': 1000,  # Maximum number of iterations
        'use_weights': False,  # Whether to use edge weights
        'weight_key': 'weight',  # Edge attribute key for weights
        'singleton_clusters': False,  # Whether to include singleton clusters
        'min_cluster_size': 2,  # Minimum cluster size to retain
        },
}