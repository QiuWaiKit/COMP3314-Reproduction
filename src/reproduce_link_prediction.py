import numpy as np
import torch
import scipy.sparse as sp

# Import required functions from the provided nafs_tutorial.py
from nafs_tutorial import load_dataset, nafs_embed, mask_test_edges, evaluate_link_prediction, OPTIMAL_HOPS

def main():
    datasets = ['cora', 'citeseer', 'pubmed']
    modes = ['mean', 'max', 'concat']
    
    # Use the same 3 seeds as the clustering tests
    seeds = [10, 42, 99]

    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"🚀 Testing NAFS Link Prediction | Dataset: {ds.upper()}")
        print(f"{'='*70}")
        
        # 1. Load the dataset
        adj, features, labels = load_dataset(ds)
        
        # 2. Extract the original adjacency matrix (without self-loops) for final evaluation
        adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj_orig.eliminate_zeros()

        for mode in modes:
            # Get the optimal Hop for link_prediction from the paper's setting
            hop = OPTIMAL_HOPS['link_prediction'][ds][mode]
            print(f"\n--- Mode: {mode:6s} | Hop: {hop:2d} ---")
            
            auc_list = []
            ap_list = []
            
            for seed in seeds:
                # Fix the random seed to ensure the train/test edge split is identical for each run
                np.random.seed(seed)
                
                # Split edges (this function uses np.random internally)
                adj_train, _, _, _, test_edges, test_edges_false = mask_test_edges(adj)
                
                # Compute NAFS embeddings on the training graph
                Z = nafs_embed(adj_train, features, hop=hop, mode=mode)
                
                # Compute similarity matrix and evaluate
                sim = torch.sigmoid(torch.mm(Z, Z.T)).numpy()
                
                # evaluate_link_prediction prints the result and returns (auc, ap)
                auc, ap = evaluate_link_prediction(sim, adj_orig, test_edges, test_edges_false, hop, mode, ds)
                
                auc_list.append(auc * 100)
                ap_list.append(ap * 100)
                
            # Calculate average scores
            avg_auc = np.mean(auc_list)
            avg_ap = np.mean(ap_list)
            print(f"  🏆 Average Score -> AUC: {avg_auc:.2f}, AP: {avg_ap:.2f}")

if __name__ == '__main__':
    main()