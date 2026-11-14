import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

# Pure import: only import from nafs_tutorial (original NAFS), absolutely no hybrid extension
from nafs_tutorial import load_dataset, nafs_embed, cluster_accuracy

# 1. Strictly follow the optimal Hops from the paper / run.sh
PAPER_HOPS = {
    'cora':     {'mean': 17, 'max': 10, 'concat': 19},
    'citeseer': {'mean': 15, 'max': 10, 'concat': 10},
    'pubmed':   {'mean': 56, 'max': 65, 'concat': 55},
}

def evaluate_embedding(Z, labels, n_clusters, seed):
    """Given the original embedding and a seed, run KMeans and return metrics."""
    # Keep the original KMeans parameter n_init=20 unchanged
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    y_pred = kmeans.fit_predict(Z.numpy())
    acc, f1 = cluster_accuracy(labels, y_pred)
    nmi = nmi_score(labels, y_pred, average_method='arithmetic')
    ari = ari_score(labels, y_pred)
    return acc * 100, nmi * 100, ari * 100

def main():
    datasets = ['cora', 'citeseer', 'pubmed']
    modes = ['mean', 'max', 'concat']
    
    # 2. Use the default seed=10 from nafs_tutorial.py, plus two randomly picked seeds (42, 99)
    seeds = [10, 42, 99] 

    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"🚀 Testing NAFS | Dataset: {ds.upper()}")
        print(f"{'='*70}")
        
        # Load data
        print("Loading data...")
        adj, features, labels = load_dataset(ds)
        n_clusters = labels.max() + 1

        for mode in modes:
            # 3. Strictly use the Hops defined in run.sh
            hop = PAPER_HOPS[ds][mode]
            print(f"\n--- Mode: {mode:6s} | Hop: {hop:2d}  ---")
            
            # Compute original NAFS embeddings (no alpha, beta parameters at all)
            print("  Computing original NAFS embeddings...")
            Z = nafs_embed(adj, features, hop=hop, mode=mode)
            
            results = []
            # Run with 3 different seeds
            for seed in seeds:
                acc, nmi, ari = evaluate_embedding(Z, labels, n_clusters, seed)
                results.append((acc, nmi, ari))
                print(f"    Seed={seed:3d} -> Acc: {acc:.2f}, NMI: {nmi:.2f}, ARI: {ari:.2f}")
            
            # Calculate averages
            avg_acc = np.mean([r[0] for r in results])
            avg_nmi = np.mean([r[1] for r in results])
            avg_ari = np.mean([r[2] for r in results])
            
            print(f"  🏆 Average Score -> Acc: {avg_acc:.2f}, NMI: {avg_nmi:.2f}, ARI: {avg_ari:.2f}")

if __name__ == '__main__':
    main()