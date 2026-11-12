"""
NAFS-Hybrid: Hybrid Hop Scoring for Node-Adaptive Feature Smoothing
====================================================================
Extends NAFS by combining two scores per node per hop:

  1. Smoothing Score  D_i(k):  "Is this node over-smoothed?"
     - Original NAFS uses cosine similarity between X^(k) and raw X
     - Higher = node still retains unique information
     - Per-node, per-hop

  2. Utility Score   U_i(k):  "Is this hop's representation useful?"
     - Neighborhood discrimination: avg similarity to true neighbors
       minus avg similarity to random nodes
     - Higher = this hop better captures graph structure
     - Per-node, per-hop

  Combined:  S_i(k) = α * normalize(D_i(k)) + β * normalize(U_i(k))
  Weights:   w_i(k) = softmax_k(S_i(k))

α + β = 1. α=1, β=0 recovers original NAFS.
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
# Data Loading
# ============================================================

def load_dataset(name):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for n in names:
        with open(f"../data/ind.{name}.{n}", 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            objects.append(u.load())
    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx = []
    with open(f"../data/ind.{name}.test.index") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    test_idx_sorted = np.sort(test_idx)

    if name == 'citeseer':
        full_range = range(min(test_idx), max(test_idx) + 1)
        tx_ext = sp.lil_matrix((len(full_range), x.shape[1]))
        tx_ext[test_idx_sorted - min(test_idx), :] = tx
        tx = tx_ext
        ty_ext = np.zeros((len(full_range), y.shape[1]))
        ty_ext[test_idx_sorted - min(test_idx), :] = ty
        ty = ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    features = torch.FloatTensor(np.array(features.todense()))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    labels = np.argmax(labels, axis=1)

    return adj, features, labels


# ============================================================
# Adjacency Normalization
# ============================================================

def normalize_adj(adj, r=0.5):
    adj = sp.coo_matrix(adj) + sp.eye(adj.shape[0])
    degree = np.array(adj.sum(1)).flatten()

    d_left = np.power(degree, r - 1)
    d_left[np.isinf(d_left)] = 0.
    D_left = sp.diags(d_left)

    d_right = np.power(degree, -r)
    d_right[np.isinf(d_right)] = 0.
    D_right = sp.diags(d_right)

    adj_norm = D_left.dot(adj).dot(D_right).tocoo()

    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = torch.Size(adj_norm.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


# ============================================================
# Core: Utility Score
# ============================================================

def compute_utility_score(hop_features_list, adj, n_random=5):
    """
    For each node i and each hop k, compute:
      U_i(k) = mean cos_sim(X_k[i], X_k[neighbors])
             - mean cos_sim(X_k[i], X_k[random_nodes])

    This measures whether hop k successfully captures local graph structure.
    High U → neighbors look more similar than random nodes → useful.
    Low U → neighbors and random nodes look the same → hop is useless.

    Args:
      hop_features_list: list of tensors, each (n_nodes, feat_dim)
      adj: scipy sparse adjacency matrix
      n_random: how many random nodes to sample per node as baseline

    Returns: (n_nodes, hop+1) tensor of utility scores
    """
    n_nodes = hop_features_list[0].shape[0]
    device = hop_features_list[0].device

    # Build neighbor list from adj
    adj_dense = adj.tocsr()

    utilities = []
    for k, X_k in enumerate(hop_features_list):
        # Normalize features for cosine similarity
        X_norm = F.normalize(X_k, p=2, dim=1)  # (n, d)

        # --- Average similarity to neighbors ---
        # neighbor_sim[i] = mean over j in neighbors(i) of X_norm[i] · X_norm[j]
        neighbor_sim = torch.zeros(n_nodes, device=device)
        neighbor_count = torch.zeros(n_nodes, device=device)

        coo = adj.tocoo()
        rows = torch.from_numpy(coo.row.astype(np.int64))
        cols = torch.from_numpy(coo.col.astype(np.int64))
        sim = (X_norm[rows] * X_norm[cols]).sum(dim=1)  # (n_edges,)

        neighbor_sim.scatter_add_(0, rows, sim)
        neighbor_count.scatter_add_(0, rows, torch.ones(len(rows)))

        # Avoid division by zero for isolated nodes
        neighbor_count = neighbor_count.clamp(min=1)
        avg_neighbor_sim = neighbor_sim / neighbor_count

        # --- Average similarity to random nodes (baseline) ---
        # Sample random node pairs
        rand_i = torch.randint(0, n_nodes, (n_nodes * n_random,))
        rand_j = torch.randint(0, n_nodes, (n_nodes * n_random,))
        rand_sim = (X_norm[rand_i] * X_norm[rand_j]).sum(dim=1)
        # Average per node i
        avg_rand_sim = torch.zeros(n_nodes, device=device)
        avg_rand_sim.scatter_add_(0, rand_i, rand_sim)
        avg_rand_sim /= n_random

        # Utility = neighbor similarity minus random baseline
        U_i_k = avg_neighbor_sim - avg_rand_sim
        utilities.append(U_i_k)

    return torch.stack(utilities, dim=1)  # (n_nodes, hop+1)


# ============================================================
# Core: Hybrid Smooth Features
# ============================================================

def smooth_features_hybrid(adj, features, hop, r=0.5, alpha=0.55, beta=0.45):
    """
    Hybrid NAFS smoothing:
      D_i(k) = cosine similarity between X^(k) and raw X  (smoothing score)
      U_i(k) = neighbor discrimination score               (utility score)

      S_i(k) = α * normalize(D_i(k)) + β * normalize(U_i(k))
      w_i(k) = softmax_k(S_i(k))
      Z_i    = sum_k w_i(k) * X^(k)_i

    α=1, β=0 → original NAFS.
    """
    # --- Multi-hop propagation ---
    adj_norm = normalize_adj(adj, r)
    hop_features = [features]
    for _ in range(hop):
        hop_features.append(torch.spmm(adj_norm, hop_features[-1]))

    # --- Smoothing score: cosine similarity vs raw features ---
    norm_x = torch.norm(features, p=2, dim=1).clamp(min=1e-10)
    smoothing_scores = []
    for xf in hop_features:
        norm_k = torch.norm(xf, p=2, dim=1).clamp(min=1e-10)
        cos_sim = (features * xf).sum(dim=1) / (norm_x * norm_k)
        smoothing_scores.append(cos_sim)
    D = torch.stack(smoothing_scores, dim=1)  # (n_nodes, hop+1)

    # --- Utility score: neighbor discrimination ---
    U = compute_utility_score(hop_features, adj)  # (n_nodes, hop+1)

    # --- Normalize each score across hops (per node) ---
    def normalize_across_hops(score_matrix):
        # Min-max normalize per node across hops
        s_min = score_matrix.min(dim=1, keepdim=True).values
        s_max = score_matrix.max(dim=1, keepdim=True).values
        rng = (s_max - s_min).clamp(min=1e-10)
        return (score_matrix - s_min) / rng

    D_norm = normalize_across_hops(D)
    U_norm = normalize_across_hops(U)

    # --- Combine scores ---
    S = alpha * D_norm + beta * U_norm
    weight = F.softmax(S, dim=1)

    # --- Weighted sum ---
    stacked = torch.stack(hop_features, dim=0)
    Z = torch.einsum('ik,kif->if', weight, stacked)

    return Z


def nafs_hybrid_embed(adj, features, hop,
                       r_list=(0, 0.1, 0.2, 0.3, 0.4, 0.5),
                       mode='mean', alpha=0.55, beta=0.45):
    """
    Full hybrid NAFS pipeline.
    α = weight on smoothing score (original NAFS)
    β = weight on utility score (new)
    """
    embeddings = []
    for r in r_list:
        Z_r = smooth_features_hybrid(adj, features, hop, r=r,
                                       alpha=alpha, beta=beta)
        embeddings.append(Z_r)

    if mode == 'mean':
        result = torch.stack(embeddings, dim=0).mean(dim=0)
    elif mode == 'max':
        result = torch.stack(embeddings, dim=0).max(dim=0).values
    elif mode == 'concat':
        result = torch.cat(embeddings, dim=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return result


# ============================================================
# Evaluation Helpers
# ============================================================

def cluster_accuracy(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    l2 = list(set(y_pred))
    cost = np.zeros((len(l1), len(l2)), dtype=int)
    for i, c1 in enumerate(l1):
        true_idx = [idx for idx, e in enumerate(y_true) if e == c1]
        for j, c2 in enumerate(l2):
            pred_idx = [idx for idx in true_idx if y_pred[idx] == c2]
            cost[i][j] = len(pred_idx)
    m = Munkres()
    indexes = m.compute(cost.__neg__().tolist())
    new_pred = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [idx for idx, e in enumerate(y_pred) if e == c2]
        new_pred[ai] = c
    acc = metrics.accuracy_score(y_true, new_pred)
    f1 = metrics.f1_score(y_true, new_pred, average='macro')
    return acc, f1


def mask_test_edges(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_triu = sp.triu(adj)
    edges = np.array(adj_triu.nonzero()).T
    edges_all = np.array(adj.nonzero()).T
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    all_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_idx)
    val_idx = all_idx[:num_val]
    test_idx = all_idx[num_val:(num_val + num_test)]
    train_idx = np.setdiff1d(np.arange(edges.shape[0]),
                              np.hstack([val_idx, test_idx]))
    val_edges = edges[val_idx]
    test_edges = edges[test_idx]
    train_edges = edges[train_idx]

    def ismember(a, b):
        rows_close = np.all(np.round(a - b[:, None], 5) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if i == j or ismember([i, j], edges_all):
            continue
        if test_edges_false and (ismember([i, j], np.array(test_edges_false)) or
                                  ismember([j, i], np.array(test_edges_false))):
            continue
        test_edges_false.append([i, j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if i == j:
            continue
        if ismember([i, j], train_edges) or ismember([j, i], train_edges):
            continue
        if val_edges_false and (ismember([i, j], np.array(val_edges_false)) or
                                 ismember([j, i], np.array(val_edges_false))):
            continue
        val_edges_false.append([i, j])

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])),
                              shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


# ============================================================
# Main
# ============================================================

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


def run_clustering(dataset='cora', hop=None, mode='mean',
                   alpha=0.55, beta=0.45, seed=63):
    task = 'clustering'
    if hop is None:
        hop = OPTIMAL_HOPS[task][dataset][mode]

    print(f"\n{'='*65}")
    print(f"  [Hybrid] Clustering | {dataset} | {mode} | hop={hop}")
    print(f"  alpha={alpha} (smoothing)  beta={beta} (utility)")
    print(f"{'='*65}")

    adj, features, labels = load_dataset(dataset)
    n_clusters = labels.max() + 1
    print(f"Nodes: {features.shape[0]}, Features: {features.shape[1]}, Classes: {n_clusters}")

    print("Computing hybrid NAFS embedding...")
    Z = nafs_hybrid_embed(adj, features, hop=hop, mode=mode,
                          alpha=alpha, beta=beta)
    print(f"Embedding dim: {Z.shape[1]}")

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    y_pred = kmeans.fit_predict(Z.numpy())

    acc, f1 = cluster_accuracy(labels, y_pred)
    nmi = nmi_score(labels, y_pred, average_method='arithmetic')
    ari = ari_score(labels, y_pred)
    print(f"  Acc={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  F1={f1:.4f}")
    return acc, nmi, ari


def run_link_prediction(dataset='cora', hop=None, mode='mean',
                        alpha=0.55, beta=0.45, seed=63):
    task = 'link_prediction'
    if hop is None:
        hop = OPTIMAL_HOPS[task][dataset][mode]

    print(f"\n{'='*65}")
    print(f"  [Hybrid] Link Prediction | {dataset} | {mode} | hop={hop}")
    print(f"  alpha={alpha} (smoothing)  beta={beta} (utility)")
    print(f"{'='*65}")

    np.random.seed(seed)
    adj, features, labels = load_dataset(dataset)

    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()

    adj_train, _, _, _, test_edges, test_edges_false = mask_test_edges(adj)

    print(f"Nodes: {features.shape[0]}, Features: {features.shape[1]}")
    print("Computing hybrid NAFS embedding on training graph...")
    Z = nafs_hybrid_embed(adj_train, features, hop=hop, mode=mode,
                          alpha=alpha, beta=beta)
    print(f"Embedding dim: {Z.shape[1]}")

    sim = torch.sigmoid(torch.mm(Z, Z.T)).numpy()

    preds, pos = [], []
    for e in test_edges:
        preds.append(sim[e[0], e[1]])
        pos.append(1)
    preds_neg, neg = [], []
    for e in test_edges_false:
        preds_neg.append(sim[e[0], e[1]])
        neg.append(0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    auc = roc_auc_score(labels_all, preds_all)
    ap = average_precision_score(labels_all, preds_all)
    print(f"  AUC={auc:.4f}  AP={ap:.4f}")
    return auc, ap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NAFS-Hybrid: dual-score hop weighting")
    parser.add_argument('--task', type=str, default='clustering',
                        choices=['clustering', 'link_prediction'])
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--mode', type=str, default='mean',
                        choices=['mean', 'max', 'concat'])
    parser.add_argument('--hop', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=0.55,
                        help='Weight on smoothing score D (default 0.55)')
    parser.add_argument('--beta', type=float, default=0.45,
                        help='Weight on utility score U  (default 0.45)')
    parser.add_argument('--seed', type=int, default=63)
    args = parser.parse_args()

    if args.task == 'clustering':
        run_clustering(dataset=args.dataset, hop=args.hop, mode=args.mode,
                       alpha=args.alpha, beta=args.beta, seed=args.seed)
    else:
        run_link_prediction(dataset=args.dataset, hop=args.hop, mode=args.mode,
                            alpha=args.alpha, beta=args.beta, seed=args.seed)
