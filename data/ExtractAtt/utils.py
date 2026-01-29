import os
import pickle
from pathlib import Path
from tqdm import tqdm
import networkx as nx
from datetime import datetime, timedelta

def load_graph(path):
    """Loads a graph from a pickle file."""
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G

def save_graph(G, savepath, measure):
    """Saves a graph to a pickle file."""
    savefile = os.path.join(savepath, f'graph_with_{measure}.pkl')
    with open(savefile, 'wb') as f:
        pickle.dump(G, f)

def merge_graph(originG, result_path, find_regex="*.pkl"):
    """Merges attributes from multiple pickled graphs into the original graph."""
    newG = originG.copy()
    path = Path(result_path)
    pkl_files = list(path.glob(find_regex))
    
    for pkl in tqdm(pkl_files, desc="Merging graphs from .pkl files", ncols=50):
        try:
            with open(pkl, 'rb') as f:
                G = pickle.load(f)
            
            # Ensure topology matches before composing
            if G.number_of_nodes() == newG.number_of_nodes():
                newG = nx.compose(newG, G)
        except Exception as e:
            print(f"Error merging graph from {pkl}: {e}")
            continue
            
    return newG

def time_calc(start, end):
    """Calculates elapsed time in hours, minutes, and seconds."""
    elapsed = timedelta(seconds=end - start)
    total = int(elapsed.total_seconds())
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds