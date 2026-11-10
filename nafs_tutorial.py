"""
NAFS (Node-Adaptive Feature Smoothing) — Clean Implementation
=============================================================
Core idea:
  1. Propagate node features X through the adjacency matrix A for K steps,
     obtaining X^(0), X^(1), ..., X^(K)
       - X^(0) = X (raw features)
       - X^(k) = A_norm · X^(k-1) (each step adds one more hop of neighbors)
  2. For each node, compute an "attention weight" that decides how much
     each hop's feature contributes:
       - Use cosine similarity between X^(k) and the raw X as importance
       - More similar → less smoothed → more unique information → higher weight
  3. Weighted sum gives the final node embedding Z
  4. Run with multiple normalization parameters r, then combine via
     mean / max / concat

Downstream tasks:
  - Clustering: KMeans on Z, evaluated by Acc / NMI / ARI / F1
  - Link prediction: sigmoid(Z · Z^T) as edge score, evaluated by AUC / AP
"""

import argparse
import pickle as pkl

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
from munkres import Munkres


# ============================================================
# Part 1: Data Loading (Planetoid format: Cora / Citeseer / Pubmed)
# ============================================================

def load_dataset(name):
    """Load a Planetoid-format graph dataset (cora / citeseer / pubmed)."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for n in names:
        with open(f"../data/ind.{name}.{n}", 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            objects.append(u.load())
    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # Read test split index
    test_idx = []
    with open(f"../data/ind.{name}.test.index") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    test_idx_sorted = np.sort(test_idx)

    # Citeseer has isolated nodes — pad with zero rows
    if name == 'citeseer':
        full_range = range(min(test_idx), max(test_idx) + 1)
        tx_ext = sp.lil_matrix((len(full_range), x.shape[1]))
        tx_ext[test_idx_sorted - min(test_idx), :] = tx
        tx = tx_ext
        ty_ext = np.zeros((len(full_range), y.shape[1]))
        ty_ext[test_idx_sorted - min(test_idx), :] = ty
        ty = ty_ext

    # Stack all node features and labels
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    features = torch.FloatTensor(np.array(features.todense()))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    labels = np.argmax(labels, axis=1)  # one-hot → class index

    return adj, features, labels


# ============================================================
# Part 2: Adjacency Normalization
# ============================================================

def normalize_adj(adj, r=0.5):
    """
    Generalized symmetric normalization: D^(r-1) · A' · D^(-r)
      - r=0.5 → standard GCN normalization (D^-1/2 · A' · D^-1/2)
      - r=0.0 → random walk transition matrix (D^-1 · A')
      - r=1.0 → reverse random walk (A' · D^-1)

    Different r values weight edges differently, capturing graph structure
    at different scales.
    """
    adj = sp.coo_matrix(adj) + sp.eye(adj.shape[0])  # add self-loops
    degree = np.array(adj.sum(1)).flatten()

    # Left: D^(r-1)
    d_left = np.power(degree, r - 1)
    d_left[np.isinf(d_left)] = 0.
    D_left = sp.diags(d_left)

    # Right: D^(-r)
    d_right = np.power(degree, -r)
    d_right[np.isinf(d_right)] = 0.
    D_right = sp.diags(d_right)

    adj_norm = D_left.dot(adj).dot(D_right).tocoo()

    # Convert to torch sparse tensor
    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = torch.Size(adj_norm.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


# ============================================================
# Part 3: NAFS Core — Node-Adaptive Feature Smoothing
# ============================================================

def smooth_features(adj, features, hop, r=0.5):
    """
    Run the full smoothing + adaptive weighting for a single (adj_norm, r):

    Steps:
      1. Starting from X^(0)=X, propagate hop times to get X^(0)~X^(hop)
      2. Compute cosine similarity between each hop's feature and raw X
      3. Softmax over similarities → attention weight w_i(k) per node per hop
      4. Weighted sum: Z_i = sum_k w_i(k) * X^(k)_i

    Returns: Z, shape (n_nodes, feat_dim)
    """
    # --- Step 1: Multi-hop propagation ---
    adj_norm = normalize_adj(adj, r)
    hop_features = [features]  # hop_features[0] = raw features
    for _ in range(hop):
        hop_features.append(torch.spmm(adj_norm, hop_features[-1]))

    # --- Step 2: Cosine similarity (each hop vs raw features) ---
    norm_x = torch.norm(features, p=2, dim=1).clamp(min=1e-10)

    similarities = []
    for xf in hop_features:
        norm_k = torch.norm(xf, p=2, dim=1).clamp(min=1e-10)
        cos_sim = (features * xf).sum(dim=1) / (norm_x * norm_k)
        similarities.append(cos_sim)

    # --- Step 3: Softmax → attention weights ---
    # Stack → (n_nodes, hop+1), softmax along the hop dimension
    weight = F.softmax(torch.stack(similarities, dim=1), dim=1)

    # --- Step 4: Weighted sum (vectorized via einsum) ---
    # Z[i,f] = sum_k weight[i,k] * hop_features[k][i,f]
    stacked = torch.stack(hop_features, dim=0)  # (hop+1, n_nodes, feat_dim)
    Z = torch.einsum('ik,kif->if', weight, stacked)  # (n_nodes, feat_dim)

    return Z


def nafs_embed(adj, features, hop, r_list=(0, 0.1, 0.2, 0.3, 0.4, 0.5), mode='mean'):
    """
    Full NAFS pipeline: multi-r smoothing + feature ensemble.

    Args:
      hop:     maximum smoothing step K
      r_list:  list of normalization parameters, each producing different smoothed features
      mode:    ensemble strategy, one of:
                 - 'mean'   → element-wise average
                 - 'max'    → element-wise maximum
                 - 'concat' → concatenate along feature dimension
    """
    embeddings = []

    for r in r_list:
        Z_r = smooth_features(adj, features, hop, r=r)
        embeddings.append(Z_r)

    if mode == 'mean':
        # Element-wise average: Z = (Z_0 + Z_1 + ... + Z_R) / R
        result = torch.stack(embeddings, dim=0).mean(dim=0)

    elif mode == 'max':
        # Element-wise maximum: Z[i,f] = max_r Z_r[i,f]
        result = torch.stack(embeddings, dim=0).max(dim=0).values

    elif mode == 'concat':
        # Concatenate: [Z_0 | Z_1 | ... | Z_R], feature dim becomes R times
        result = torch.cat(embeddings, dim=1)

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from mean / max / concat.")

    return result


# ============================================================
# Part 4: Clustering Evaluation
# ============================================================

def cluster_accuracy(y_true, y_pred):
    """
    Use the Hungarian algorithm (Munkres) to find the best matching between
    predicted clusters and ground-truth labels, then compute ACC / F1.
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    l2 = list(set(y_pred))

    # Build cost matrix
    cost = np.zeros((len(l1), len(l2)), dtype=int)
    for i, c1 in enumerate(l1):
        true_idx = [idx for idx, e in enumerate(y_true) if e == c1]
        for j, c2 in enumerate(l2):
            pred_idx = [idx for idx in true_idx if y_pred[idx] == c2]
            cost[i][j] = len(pred_idx)

    # Hungarian matching
    m = Munkres()
    indexes = m.compute(cost.__neg__().tolist())

    # Remap predicted labels
    new_pred = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [idx for idx, e in enumerate(y_pred) if e == c2]
        new_pred[ai] = c

    acc = metrics.accuracy_score(y_true, new_pred)
    f1 = metrics.f1_score(y_true, new_pred, average='macro')
    return acc, f1


def evaluate_clustering(y_true, y_pred, hop, mode, dataset):
    """Run a full clustering evaluation and print all metrics."""
    acc, f1 = cluster_accuracy(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(f"[{dataset:8s} | {mode:6s} | hop={hop:2d}] "
          f"Acc={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  F1={f1:.4f}")
    return acc, nmi, ari, f1


# ============================================================
# Part 5: Link Prediction — Edge Splitting & Evaluation
# ============================================================

def mask_test_edges(adj):
    """
    Split edges into train / validation / test sets.
    10% of edges → test, 5% → validation, rest → train.
    Also sample the same number of negative (non-existent) edges for val/test.

    Returns:
      adj_train: adjacency matrix after removing val/test edges
      train_edges, val_edges, val_edges_false, test_edges, test_edges_false
    """
    # Remove diagonal (self-loops)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    adj_triu = sp.triu(adj)
    edges = np.array(adj_triu.nonzero()).T  # upper-triangle edges
    edges_all = np.array(adj.nonzero()).T

    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    # Shuffle and split
    all_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_idx)
    val_idx = all_idx[:num_val]
    test_idx = all_idx[num_val:(num_val + num_test)]
    train_idx = np.delete(edges.shape[0] and np.arange(edges.shape[0]),
                           np.hstack([val_idx, test_idx]), axis=0)

    val_edges = edges[val_idx]
    test_edges = edges[test_idx]
    train_edges = edges[np.setdiff1d(np.arange(edges.shape[0]),
                                      np.hstack([val_idx, test_idx]))]

    # Helper: check if an edge exists
    def ismember(a, b):
        rows_close = np.all(np.round(a - b[:, None], 5) == 0, axis=-1)
        return np.any(rows_close)

    # Sample negative edges for test
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if i == j:
            continue
        if ismember([i, j], edges_all):
            continue
        if test_edges_false and (ismember([i, j], np.array(test_edges_false)) or
                                  ismember([j, i], np.array(test_edges_false))):
            continue
        test_edges_false.append([i, j])

    # Sample negative edges for validation
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if i == j:
            continue
        if ismember([i, j], train_edges) or ismember([j, i], train_edges):
            continue
        if ismember([i, j], val_edges) or ismember([j, i], val_edges):
            continue
        if val_edges_false and (ismember([i, j], np.array(val_edges_false)) or
                                 ismember([j, i], np.array(val_edges_false))):
            continue
        val_edges_false.append([i, j])

    # Rebuild train adjacency (symmetric)
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])),
                              shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def evaluate_link_prediction(sim, adj_orig, edges_pos, edges_neg, hop, mode, dataset):
    """Compute AUC and AP from a similarity matrix Z · Z^T."""
    preds, pos = [], []
    for e in edges_pos:
        preds.append(sim[e[0], e[1]])
        pos.append(1)

    preds_neg, neg = [], []
    for e in edges_neg:
        preds_neg.append(sim[e[0], e[1]])
        neg.append(0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    auc = roc_auc_score(labels_all, preds_all)
    ap = average_precision_score(labels_all, preds_all)

    print(f"[{dataset:8s} | {mode:6s} | hop={hop:2d}] "
          f"AUC={auc:.4f}  AP={ap:.4f}")
    return auc, ap


# ============================================================
# Part 6: Main Pipeline
# ============================================================

# Optimal hop values from the paper / run.sh (per dataset × mode × task)
OPTIMAL_HOPS = {
    'clustering': {
        'cora':     {'mean': 17, 'max': 10, 'concat': 19},
        'citeseer': {'mean': 15, 'max': 10, 'concat': 10},
        'pubmed':   {'mean': 56, 'max': 65, 'concat': 55},
    },
    'link_prediction': {
        'cora':     {'mean': 9, 'max': 8, 'concat': 10},
        'citeseer': {'mean': 4, 'max': 4, 'concat': 12},
        'pubmed':   {'mean': 6, 'max': 5, 'concat': 6},
    },
}


def run_clustering(dataset='cora', hop=None, mode='mean', seed=42):
    """Run NAFS + KMeans clustering."""
    task = 'clustering'
    if hop is None:
        hop = OPTIMAL_HOPS[task][dataset][mode]

    print(f"\n{'='*60}")
    print(f"  Task: Clustering  |  Dataset: {dataset}  |  Mode: {mode}  |  Hop: {hop}")
    print(f"{'='*60}")

    adj, features, labels = load_dataset(dataset)
    n_clusters = labels.max() + 1
    print(f"Nodes: {features.shape[0]}, Features: {features.shape[1]}, "
          f"Classes: {n_clusters}")

    print("Computing NAFS embedding...")
    Z = nafs_embed(adj, features, hop=hop, mode=mode)
    print(f"Embedding dim: {Z.shape[1]}")

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    y_pred = kmeans.fit_predict(Z.numpy())

    evaluate_clustering(labels, y_pred, hop, mode, dataset)


def run_link_prediction(dataset='cora', hop=None, mode='mean', seed=42):
    """Run NAFS + link prediction (AUC / AP)."""
    task = 'link_prediction'
    if hop is None:
        hop = OPTIMAL_HOPS[task][dataset][mode]

    print(f"\n{'='*60}")
    print(f"  Task: Link Prediction  |  Dataset: {dataset}  |  Mode: {mode}  |  Hop: {hop}")
    print(f"{'='*60}")

    np.random.seed(seed)

    adj, features, labels = load_dataset(dataset)

    # Keep original adj (no self-loops) for evaluation
    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()

    # Split edges into train / val / test
    adj_train, _, _, _, test_edges, test_edges_false = mask_test_edges(adj)

    print(f"Nodes: {features.shape[0]}, Features: {features.shape[1]}")
    print("Computing NAFS embedding on training graph...")
    Z = nafs_embed(adj_train, features, hop=hop, mode=mode)
    print(f"Embedding dim: {Z.shape[1]}")

    # Similarity matrix: sigmoid(Z · Z^T)
    sim = torch.sigmoid(torch.mm(Z, Z.T)).numpy()

    evaluate_link_prediction(sim, adj_orig, test_edges, test_edges_false,
                             hop, mode, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NAFS clean implementation")
    parser.add_argument('--task', type=str, default='clustering',
                        choices=['clustering', 'link_prediction'],
                        help='Downstream task')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--hop', type=int, default=None,
                        help='Max smoothing step K (auto: paper optimal value)')
    parser.add_argument('--mode', type=str, default='mean',
                        choices=['mean', 'max', 'concat'], help='Feature ensemble strategy')
    parser.add_argument('--seed', type=int, default=63, help='Random seed')
    args = parser.parse_args()

    if args.task == 'clustering':
        run_clustering(dataset=args.dataset, hop=args.hop, mode=args.mode, seed=args.seed)
    else:
        run_link_prediction(dataset=args.dataset, hop=args.hop, mode=args.mode, seed=args.seed)
