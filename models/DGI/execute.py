"""
DGI Training Script for Inductive Learning on Multiple Graphs

This script trains a DGI model in an inductive setting where:
- Training is done on a set of graphs (not a single graph)
- Each graph is processed independently
- The model learns to generalize to unseen graphs
- No train/val/test masks needed (entire graphs are used)
"""

import torch
import torch.nn as nn
import yaml
import os
from torch_geometric.loader import DataLoader

from models import DGI, LogReg
from utils.graph_utils import shuffle_node_features

from torch_geometric.data import Data
import networkx as nx
from utils.graph_utils import nx_to_pyg_data


def train_dgi_epoch(model, loader, optimizer, criterion, device):
    """Train DGI for one epoch on multiple graphs."""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Shuffle features for negative sampling
        shuf_fts = shuffle_node_features(batch.x)
        
        # Create labels: 1 for real, 0 for fake
        num_nodes = batch.x.size(0)
        lbl_1 = torch.ones(num_nodes, 1, device=device)
        lbl_2 = torch.zeros(num_nodes, 1, device=device)
        lbl = torch.cat((lbl_1, lbl_2), 0)
        
        logits = model(batch.x, shuf_fts, batch.edge_index, batch.batch, None, None)
        
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def extract_embeddings(model, loader, device):
    """Extract embeddings for all graphs in the loader."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeds, _ = model.embed(batch.x, batch.edge_index, batch.batch)
            
            # Get graph-level embeddings (mean pooling per graph)
            from torch_geometric.nn import global_mean_pool
            graph_embeds = global_mean_pool(embeds, batch.batch)
            
            all_embeddings.append(graph_embeds)
            if hasattr(batch, 'y'):
                # Assuming y is graph-level label
                all_labels.append(batch.y)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else None
    
    return embeddings, labels


def run_training(config_path=None):
    # Load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/DGI.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size = config['batch_size']
    nb_epochs = config['nb_epochs']
    patience = config['patience']
    lr = config['lr']
    l2_coef = config['l2_coef']
    hid_units = config['hid_units']
    nonlinearity = config['nonlinearity']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # ============================================
    # Load your graph data here
    # ============================================
    # Example: You should replace this with your own data loading
    # Expected format: List of PyG Data objects
    # Each Data object should have:
    #   - x: node features
    #   - edge_index: graph structure
    #   - y (optional): graph-level label
    
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    print("You need to implement your own data loading here.")
    print("Expected: List of torch_geometric.data.Data objects")
    print("\nExample:")
    print("  train_graphs = [data1, data2, ...]  # List of Data objects")
    print("  val_graphs = [data_val1, ...]")
    print("  test_graphs = [data_test1, ...]")
    print("="*60 + "\n")
    
    # Example: Generate some random graphs
    train_graphs = []
    val_graphs = []
    test_graphs = []
    
    # ============================================
    # Load graph
    # ============================================

    with open(config['DATA_PATH'], 'rb') as f:
        train_graphs = pkl.load(f)


    # for i in range(100): 
    #     G = nx.erdos_renyi_graph(n=20, p=0.3)
    #     # Add dummy features
    #     import pandas as pd
    #     features_df = pd.DataFrame({
    #         'feature_1': [G.degree(n) for n in G.nodes()],
    #         'feature_2': [nx.clustering(G, n) for n in G.nodes()],
    #     }, index=list(G.nodes()))
        
    #     data = nx_to_pyg_data(G, features_df)
    #     data.y = torch.tensor([i % 2], dtype=torch.long)  # Dummy binary label
    #     train_graphs.append(data)
    
    # print(f"Loaded {len(train_graphs)} training graphs")
    # print(f"Loaded {len(val_graphs)} validation graphs")
    # print(f"Loaded {len(test_graphs)} test graphs")
    
    # Create DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # Get feature size from first graph
    ft_size = train_graphs[0].x.size(1)
    
    print(f"\nFeature dimension: {ft_size}")
    print(f"Hidden units: {hid_units}")
    
    # Initialize model
    model = DGI(ft_size, hid_units, nonlinearity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING DGI")
    print("="*60)
    
    best_loss = float('inf')
    best_epoch = 0
    cnt_wait = 0
    
    for epoch in range(nb_epochs):
        train_loss = train_dgi_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                shuf_fts = shuffle_node_features(batch.x)
                
                num_nodes = batch.x.size(0)
                lbl_1 = torch.ones(num_nodes, 1, device=device)
                lbl_2 = torch.zeros(num_nodes, 1, device=device)
                lbl = torch.cat((lbl_1, lbl_2), 0)
                
                logits = model(batch.x, shuf_fts, batch.edge_index, batch.batch, None, None)
                loss = criterion(logits, lbl)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Early stopping based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi_inductive.pkl')
        else:
            cnt_wait += 1
        
        if cnt_wait >= patience:
            print(f'\nEarly stopping at epoch {epoch}')
            break
    
    print(f'\nBest epoch: {best_epoch}, Best val loss: {best_loss:.4f}')
    
    # Load best model
    print("\n" + "="*60)
    print("EXTRACTING EMBEDDINGS")
    print("="*60)
    
    model.load_state_dict(torch.load('best_dgi_inductive.pkl'))
    
    # Extract embeddings for all sets
    train_embeds, train_labels = extract_embeddings(model, train_loader, device)
    val_embeds, val_labels = extract_embeddings(model, val_loader, device)
    test_embeds, test_labels = extract_embeddings(model, test_loader, device)
    
    print(f"Train embeddings shape: {train_embeds.shape}")
    print(f"Val embeddings shape: {val_embeds.shape}")
    print(f"Test embeddings shape: {test_embeds.shape}")
    
    # Evaluate with logistic regression (if labels available)
    if train_labels is not None and test_labels is not None:
        print("\n" + "="*60)
        print("DOWNSTREAM EVALUATION")
        print("="*60)
        
        nb_classes = int(train_labels.max().item()) + 1
        xent = nn.CrossEntropyLoss()
        
        accs = []
        for run in range(10):
            log = LogReg(hid_units, nb_classes).to(device)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            
            # Train
            for _ in range(200):
                log.train()
                opt.zero_grad()
                logits = log(train_embeds)
                loss = xent(logits, train_labels)
                loss.backward()
                opt.step()
            
            # Test
            log.eval()
            with torch.no_grad():
                logits = log(test_embeds)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == test_labels).float().mean().item()
                accs.append(acc * 100)
        
        import numpy as np
        accs = np.array(accs)
        print(f"Test Accuracy: {accs.mean():.2f}% ± {accs.std():.2f}%")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    run_training()
