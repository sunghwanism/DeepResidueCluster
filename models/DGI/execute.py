import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import yaml
import os
from models import DGI, LogReg
from utils import process

# Load config
config_path = os.path.join(os.path.dirname(__file__), '../../config/DGI.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

dataset = config['dataset']
batch_size = config['batch_size']
nb_epochs = config['nb_epochs']
patience = config['patience']
lr = config['lr']
l2_coef = config['l2_coef']
drop_prob = config['drop_prob']
hid_units = config['hid_units']
sparse = config['sparse']
nonlinearity = config['nonlinearity']

adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

# Convert features to tensor
features = torch.FloatTensor(features[np.newaxis])
features = features.squeeze(0) # (N, F)

labels = torch.FloatTensor(labels[np.newaxis]).squeeze(0)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# Convert adj to edge_index for PyG
adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
if sp.isspmatrix(adj):
    adj = adj.tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
else:
    # If dense
    adj = sp.coo_matrix(adj)
    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)

model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    edge_index = edge_index.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[idx, :]

    # Labels: 1 for real, 0 for fake
    lbl_1 = torch.ones(nb_nodes, 1)
    lbl_2 = torch.zeros(nb_nodes, 1)
    lbl = torch.cat((lbl_1, lbl_2), 0)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    # logits shape: (2*N, 1)
    logits = model(features, shuf_fts, edge_index, None, None, None) 

    loss = b_xent(logits, lbl)

    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))

embeds, _ = model.embed(features, edge_index, None)
train_embs = embeds[idx_train]
val_embs = embeds[idx_val]
test_embs = embeds[idx_test]

train_lbls = torch.argmax(labels[idx_train], dim=1)
val_lbls = torch.argmax(labels[idx_val], dim=1)
test_lbls = torch.argmax(labels[idx_test], dim=1)

tot = torch.zeros(1)
if torch.cuda.is_available():
    tot = tot.cuda()

accs = []

for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    if torch.cuda.is_available():
        log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    if torch.cuda.is_available():
        best_acc = best_acc.cuda()
    
    # Train logistic regression
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        
        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print('Average accuracy:', tot / 50)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())
