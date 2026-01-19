import torch
import torch.nn as nn
import yaml
import os
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import yaml
import os
import numpy as np
from torch_geometric.loader import DataLoader

from models.DGI.models.dgi import DGI
from models.DGI.models.logreg import LogReg
from utils.graph_utils import shuffle_node_features
from utils.graph_utils import nx_to_pyg_data

from torch_geometric.data import Data
import networkx as nx


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


def run_training(config, train_loader, val_loader, test_loader):
    # Use config object (Namespace)
    # config is already a Namespace-like object (argparse.Namespace)
    
    batch_size = config.batch_size
    nb_epochs = config.nb_epochs
    patience = config.patience
    lr = config.lr
    l2_coef = config.l2_coef
    hid_units = config.hid_units
    nonlinearity = config.nonlinearity
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Check if loaders are provided
    if not train_loader:
        raise ValueError("train_loader is None")

    # Get feature size from first batch
    first_batch = next(iter(train_loader))
    ft_size = first_batch.x.size(1)
    
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
    
    save_path = 'best_dgi_inductive.pkl' # Could be parameterized
    
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
        
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Early stopping based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
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
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    else:
        print("Warning: Best model file not found, using current model state.")
    
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
        
        accs = np.array(accs)
        print(f"Test Accuracy: {accs.mean():.2f}% ± {accs.std():.2f}%")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    # run_training() needs arguments now, so direct execution without correct context is tough.
    pass
