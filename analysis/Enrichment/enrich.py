import os
import sys
import ast
import json
from collections import Counter
from typing import Dict, List, Set, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import poisson

def _run_permutation_test(
    cluster_name: str, 
    cluster_nodes: List[str], 
    node_to_idx_map: Dict[str, int], 
    mutation_counts: np.ndarray, 
    strata_labels: np.ndarray, 
    global_pool: Dict[str, np.ndarray], 
    min_mut: int, 
    min_node: int,
    n_permutations: int,
    chunk_size=5000,
) -> Optional[Dict[str, Any]]:
    """
    Worker function to perform a stratified permutation test for a single cluster.
    """
    # Filter nodes that exist in our dataset
    valid_indices = [node_to_idx_map[n] for n in cluster_nodes if n in node_to_idx_map]
    
    if not valid_indices:
        return None

    cluster_indices = np.array(valid_indices)
    obs_sum = np.sum(mutation_counts[cluster_indices])
    cluster_size = len(cluster_indices)

    # Pre-filtering criteria
    if obs_sum < min_mut or cluster_size < min_node:
        return None

    # Identify the composition of the cluster (Stratification)
    # e.g., {'ALA_High': 2, 'GLY_Low': 1}
    cluster_strata_counts = Counter(strata_labels[cluster_indices])
    
    strata_keys = list(cluster_strata_counts.keys())
    strata_sizes = list(cluster_strata_counts.values())
    
    # Initialize null distribution container
    null_sums = np.zeros(n_permutations)
    rng = np.random.default_rng()

    try:
        for i, strata_key in enumerate(strata_keys):
            pool = global_pool.get(strata_key)
            if pool is None or len(pool) == 0:
                return None
            
            size = strata_sizes[i]
            stratum_mutation_counts = mutation_counts[pool]
            pool_len = len(pool)
            
            # Divide n_permutations into smaller chunks
            for start_idx in range(0, n_permutations, chunk_size):
                end_idx = min(start_idx + chunk_size, n_permutations)
                current_chunk_len = end_idx - start_idx
                
                # 1. Generate selections for this chunk only
                # We use rng.choice with replace=False row by row
                chunk_selections = np.array([
                    rng.choice(pool_len, size=size, replace=False) 
                    for _ in range(current_chunk_len)
                ])
                
                # 2. Map indices back to mutation counts and sum
                # chunk_selections contains indices relative to the 'pool'
                chunk_sums = np.sum(stratum_mutation_counts[chunk_selections], axis=1)
                
                # 3. Accumulate results into the global null_sums
                null_sums[start_idx:end_idx] += chunk_sums

    except ValueError as e:
        print(f"[Error] Sampling failed for cluster {cluster_name}: {e}")
        return None

    exp_sum = np.mean(null_sums) if len(null_sums) > 0 else 0.0
    
    # Calculate enrichment metrics
    # Use a small epsilon for log calculation to avoid -inf
    enrichment_score = np.log((obs_sum + 1e-10) / (exp_sum + 1e-10))

    # P-value: (Number of nulls >= observed + 1) / (Total permutations + 1)
    # Using +1 for pseudo-count smoothing (standard practice)
    p_value = (1.0 + np.sum(null_sums >= obs_sum)) / (1.0 + n_permutations)

    return {
        'cluster_id': cluster_name,
        'cluster_size': cluster_size,
        'observed_sum': float(obs_sum),
        'expected_sum': float(exp_sum),
        'null>obs': int(np.sum(null_sums >= obs_sum)),
        'enrichment_score': float(enrichment_score),
        'p_value': float(p_value)
    }


def analyze_all_clusters_parallel(
    df: pd.DataFrame, 
    clusters_dict: Dict[str, List[str]], 
    min_mut: int = 5, 
    min_nodes: int = 1,
    n_permutations: int = 1_000_000, 
    n_jobs: int = -1, 
    stratify: str = 'mut+res',
    bin_col: str = 'bin_mutability',
    PDBMatching: bool = False
) -> pd.DataFrame:
    """
    Main driver to run permutation tests on all clusters in parallel.
    Handles data preparation, parallel execution, FDR correction, and ID mapping.
    """
    # 1. Prepare Data
    df = df.reset_index(drop=True)
    node_to_idx = {node: idx for idx, node in enumerate(df['node_id'])}
    mutation_counts = df['total_mutations_count'].to_numpy(dtype=np.float32)

    # 2. Define Stratification Labels
    # Use vectorized string operations for speed
    if stratify == 'mut+res':
        strata_labels = (df['residue_name'].astype(str) + "_" + df[bin_col].astype(str)).to_numpy()
    elif stratify == 'res':
        strata_labels = df['residue_name'].astype(str).to_numpy()
    elif stratify == 'mut':
        strata_labels = df[bin_col].astype(str).to_numpy()
    else:
        # No stratification: all nodes belong to group '0'
        strata_labels = np.zeros(len(df), dtype=int).astype(str)

    # Create global pools for sampling
    unique_strata = np.unique(strata_labels)
    global_pool = {s: np.where(strata_labels == s)[0] for s in unique_strata}

    # 3. Parallel Execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_permutation_test)(
            name, nodes, node_to_idx, mutation_counts, 
            strata_labels, global_pool, min_mut, min_nodes, n_permutations
        )
        for name, nodes in tqdm(clusters_dict.items(), desc="Running Permutation Tests")
    )

    # Filter None results
    results_list = [r for r in results if r is not None]
    
    if not results_list:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)

    # 4. Multiple Testing Correction (FDR Benjamini-Hochberg)
    reject, pvals_corrected, _, _ = multipletests(results_df['p_value'].values, alpha=0.05, method='fdr_bh')
    results_df['q_value'] = pvals_corrected

    # 5. Formatting & ID Mapping
    cols = ['cluster_id', 'cluster_size', 'enrichment_score', 
            'p_value', 'q_value', 'observed_sum', 'expected_sum', 'null>obs']
    
    existing_cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[existing_cols].copy()
    
    # Helper to ensure nodes are a list
    def _parse_nodes(cluster_id):
        nodes = clusters_dict.get(cluster_id, [])
        if isinstance(nodes, str):
            try:
                return list(ast.literal_eval(nodes))
            except (ValueError, SyntaxError):
                return [nodes]
        return nodes

    results_df['nodes'] = results_df['cluster_id'].apply(_parse_nodes)
    
    # Calculate number of mutated nodes within cluster
    results_df['num_mutated_node'] = results_df['nodes'].apply(
        lambda nodes: sum(mutation_counts[node_to_idx[n]] > 0 for n in nodes if n in node_to_idx)
    )

    # Map UniProt and PDB IDs
    def _map_ids(node_list):
        unique_uniprot = set()
        
        for node in node_list:
            parts = node.split('_')
            clean_id = parts[0].split('-')[0]
            unique_uniprot.add(clean_id)
                
        return unique_uniprot

    # Apply mapping efficiently
    mapped_series = results_df['nodes'].apply(lambda x: pd.Series(_map_ids(x), index=['unique_uniprot']))
    results_df = pd.concat([results_df, mapped_series], axis=1)
    results_df['num_unique_uniprot'] = results_df['unique_uniprot'].apply(len)
    
    # if PDBMatching:
    #     print("Mapping common PDB structures...")
    #     results_df['PDB'] = results_df['unique_uniprot'].apply(find_common_pdb)

    return results_df.sort_values(by='p_value', ascending=True).reset_index(drop=True)


def _run_permutation_test_pathogenicity(
    cluster_name: str, 
    cluster_nodes: List[str], 
    node_to_idx_map: Dict[str, int], 
    pathogenicity: np.ndarray, 
    strata_labels: np.ndarray, 
    global_pool: Dict[str, np.ndarray], 
    min_node: int,
    n_permutations: int,
    chunk_size: int = 5000,
    seed: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Worker function to perform a stratified permutation test for a single cluster.
    
    [Logic Flow]
    1. Calculate Observed Mean of the real cluster.
    2. Generate N random 'Null Clusters' preserving the strata composition.
    3. Calculate the mean of *each* Null Cluster.
    4. Compare Observed Mean vs. Distribution of Null Means.
    """
    
    # 1. Validate Nodes & Get Indices
    valid_indices = [node_to_idx_map[n] for n in cluster_nodes if n in node_to_idx_map]
    
    if not valid_indices:
        return None

    cluster_indices = np.array(valid_indices)
    cluster_size = len(cluster_indices)

    # Pre-filtering
    if cluster_size < min_node:
        return None

    # [Step 1] Calculate Observed Mean
    obs_mean = np.mean(pathogenicity[cluster_indices])

    cluster_strata_counts = Counter(strata_labels[cluster_indices])
    strata_keys = list(cluster_strata_counts.keys())
    strata_sizes = list(cluster_strata_counts.values())
    
    # [Step 2] Initialize container for Null Sums
    null_sums = np.zeros(n_permutations)
    rng = np.random.default_rng(seed)

    try:
        for i, strata_key in enumerate(strata_keys):
            pool = global_pool.get(strata_key)
            if pool is None or len(pool) == 0:
                return None
            
            size = strata_sizes[i]     # How many nodes to pick for this stratum
            stratum_vals = pathogenicity[pool] # Values available in the global pool
            pool_len = len(pool)
            
            # Process in chunks to save memory
            for start_idx in range(0, n_permutations, chunk_size):
                end_idx = min(start_idx + chunk_size, n_permutations)
                current_chunk_len = end_idx - start_idx
                
                # A. Select random nodes for this chunk
                # Shape: (current_chunk_len, size)
                chunk_selections = np.empty((current_chunk_len, size), dtype=int)
                for k in range(current_chunk_len):
                    chunk_selections[k] = rng.choice(pool_len, size=size, replace=False)
                
                # B. Get values and SUM them
                # axis=1 means summing across the nodes within one permutation
                current_vals = stratum_vals[chunk_selections]
                chunk_sums = np.sum(current_vals, axis=1)
                
                # C. Accumulate the sums to the specific permutation slots
                null_sums[start_idx:end_idx] += chunk_sums

    except ValueError as e:
        print(f"[Error] Sampling failed for cluster {cluster_name}: {e}")
        return None

    # [Step 3] Calculate Null Means
    null_means = null_sums / cluster_size

    # [Step 4] Compare Observed vs Null Distribution
    exp_mean = np.mean(null_means) if len(null_means) > 0 else 0.0
    
    # Enrichment Score (Log Ratio)
    epsilon = 1e-10
    enrichment_score = np.log((obs_mean + epsilon) / (exp_mean + epsilon))

    # P-value Calculation
    # +1 is for pseudo-count (standard statistical practice to avoid p=0)
    n_extreme = np.sum(null_means >= obs_mean)
    p_value = (1.0 + n_extreme) / (1.0 + n_permutations)

    return {
        'cluster_id': cluster_name,
        'cluster_size': cluster_size,
        'observed_mean': float(obs_mean),
        'expected_mean': float(exp_mean),
        'null>obs': int(n_extreme),
        'enrichment_score': float(enrichment_score),
        'p_value': float(p_value)
    }


def analyze_all_clusters_parallel_pathogenicity(
    df: pd.DataFrame, 
    clusters_dict: Dict[str, List[str]], 
    mutation_counts_col: str, 
    min_nodes: int = 1,
    n_permutations: int = 100_000, 
    n_jobs: int = -1, 
    stratify: str = 'mut+res',
    bin_col: str = 'bin_mutability'
) -> pd.DataFrame:
    """
    Main driver function to run permutation tests in parallel.
    """
    # 1. Prepare Data
    df = df.reset_index(drop=True)
    node_to_idx = {node: idx for idx, node in enumerate(df['node_id'])}
    pathogenicity = df['avg_am_pathogenicity'].to_numpy(dtype=np.float32)

    # Check for mutation counts column
    if mutation_counts_col not in df.columns:
        raise ValueError(f"Column '{mutation_counts_col}' not found in DataFrame.")
    mutation_counts = df[mutation_counts_col].to_numpy()

    # 2. Define Stratification Labels
    if stratify == 'mut+res':
        strata_labels = (df['residue_name'].astype(str) + "_" + df[bin_col].astype(str)).to_numpy()
    elif stratify == 'res':
        strata_labels = df['residue_name'].astype(str).to_numpy()
    elif stratify == 'mut':
        strata_labels = df[bin_col].astype(str).to_numpy()
    else:
        strata_labels = np.zeros(len(df), dtype=int).astype(str)
    
    # Create global pools
    unique_strata = np.unique(strata_labels)
    global_pool = {s: np.where(strata_labels == s)[0] for s in unique_strata}

    # 3. Parallel Execution
    # seed=abs(hash(name)) ensures that if you run it again, you get the same result for the same cluster
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_permutation_test_pathogenicity)(
            name, nodes, node_to_idx, pathogenicity, 
            strata_labels, global_pool, min_nodes, n_permutations, 
            chunk_size=5000,
            seed=abs(hash(name)) % (2**32) 
        )
        for name, nodes in tqdm(clusters_dict.items(), desc="Running Permutation Tests")
    )

    # Filter None results
    results_list = [r for r in results if r is not None]
    
    if not results_list:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)

    # 4. FDR Correction
    reject, pvals_corrected, _, _ = multipletests(results_df['p_value'].values, alpha=0.05, method='fdr_bh')
    results_df['q_value'] = pvals_corrected

    # 5. Formatting & Metrics
    def _parse_nodes(cluster_id):
        nodes = clusters_dict.get(cluster_id, [])
        if isinstance(nodes, str):
            try:
                return list(ast.literal_eval(nodes))
            except (ValueError, SyntaxError):
                return [nodes]
        return nodes

    results_df['nodes'] = results_df['cluster_id'].apply(_parse_nodes)
    
    # Calculate number of mutated nodes
    results_df['num_mutated_node'] = results_df['nodes'].apply(
        lambda nodes: sum(mutation_counts[node_to_idx[n]] > 0 for n in nodes if n in node_to_idx)
    )

    # Map UniProt IDs
    def _map_ids(node_list):
        unique_uniprot = set()
        for node in node_list:
            if isinstance(node, str):
                parts = node.split('_')
                if parts:
                    clean_id = parts[0].split('-')[0]
                    unique_uniprot.add(clean_id)
        return unique_uniprot

    mapped_series = results_df['nodes'].apply(lambda x: pd.Series([_map_ids(x)], index=['unique_uniprot']))
    results_df = pd.concat([results_df, mapped_series], axis=1)
    results_df['num_unique_uniprot'] = results_df['unique_uniprot'].apply(len)
    
    # Select Columns
    final_cols = [
        'cluster_id', 'cluster_size', 'num_mutated_node', 'num_unique_uniprot',
        'observed_mean', 'expected_mean', 'null>obs', 
        'enrichment_score', 'p_value', 'q_value', 
        'nodes', 'unique_uniprot'
    ]
    
    existing_cols = [c for c in final_cols if c in results_df.columns]
    
    return results_df[existing_cols].sort_values(by='p_value', ascending=True).reset_index(drop=True)