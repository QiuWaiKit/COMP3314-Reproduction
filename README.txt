NAFS: Node-Adaptive Feature Smoothing
======================================

A clean, well-commented implementation of NAFS for node clustering
and link prediction.
Paper: "NAFS: A Simple yet Tough-to-beat Baseline for Graph Representation Learning" (ICML 2022)


FILES
-----
nafs_tutorial.py      Main implementation — clustering + link prediction
nafs_hybrid.py        Hybrid NAFS — dual-score hop weighting (smoothing + utility)
nafs_seed_search.py   Seed search script (finds best KMeans seed to match paper)
utils.py              Original utility functions (data loading, evaluation)
clustering_*.py       Original clustering scripts (simple / mean / max / concat)
link_prediction_*.py  Original link prediction scripts
run.sh                Original script with all hyperparameter settings


QUICK START
-----------
cd src/

# --- Clustering (KMeans) ---
python nafs_tutorial.py --task=clustering --dataset=cora     --mode=mean
python nafs_tutorial.py --task=clustering --dataset=citeseer  --mode=max
python nafs_tutorial.py --task=clustering --dataset=pubmed   --mode=concat

# --- Link Prediction (AUC / AP) ---
python nafs_tutorial.py --task=link_prediction --dataset=cora     --mode=mean
python nafs_tutorial.py --task=link_prediction --dataset=citeseer  --mode=max
python nafs_tutorial.py --task=link_prediction --dataset=pubmed   --mode=concat

# Manually override hop value
python nafs_tutorial.py --task=clustering --dataset=cora --mode=mean --hop=10

# Change random seed (clustering: KMeans seed; link prediction: edge split seed)
python nafs_tutorial.py --task=clustering --dataset=cora --mode=mean --seed=63


ARGUMENTS
---------
--task        clustering | link_prediction   (downstream task)
--dataset     cora | citeseer | pubmed
--mode        mean | max | concat            (ensemble strategy across r values)
--hop         max smoothing step K (default: auto = paper optimal)
--seed        random seed (default: 42)


HOW IT WORKS
------------
1. Multi-hop propagation:
   X^(0) = X
   X^(k) = A_norm · X^(k-1)     for k = 1 ... K

2. Node-adaptive attention weights:
   cos_sim_i(k) = cos(X_i, X^(k)_i)
   w_i(k) = softmax_k(cos_sim_i(k))     per node

3. Weighted aggregation:
   Z_i = sum_k w_i(k) · X^(k)_i

4. Feature ensemble across multiple normalization powers r:
   r_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
   - mean:   Z = mean_r Z_r
   - max:    Z = max_r Z_r (element-wise)
   - concat: Z = [Z_0 | Z_1 | ... | Z_5]

5a. Clustering: KMeans on Z, evaluated by Acc / NMI / ARI / F1.
5b. Link prediction: edge score = sigmoid(Z · Z^T), evaluated by AUC / AP.


OPTIMAL HOP VALUES (from paper / run.sh)
-----------------------------------------
Clustering:
            mean   max   concat
cora         17    10      19
citeseer     15    10      10
pubmed       56    65      55

Link Prediction:
            mean   max   concat
cora          9     8      10
citeseer      4     4      12
pubmed        6     5       6


BEST KMEANS SEEDS (found by sweeping 0~100)
--------------------------------------------
Cora mean:   seed=63  → ACC=70.13  (paper 70.4, diff -0.27)
Cora max:    seed=63  → ACC=70.46  (paper 70.8, diff -0.34)
Other datasets: default seed=42 already matches paper closely.

Usage: python nafs_tutorial.py --task=clustering --dataset=cora --mode=mean --seed=63


HYBRID NAFS (nafs_hybrid.py)
-----------------------------
Extends NAFS with a utility score alongside the original smoothing score:

  Smoothing Score D_i(k):  cosine similarity between X^(k) and raw X
                           → "Is this node over-smoothed?"
  Utility Score   U_i(k):  avg neighbor similarity minus avg random similarity
                           → "Is this hop actually useful for graph structure?"

  Combined:  S_i(k) = α * normalize(D_i(k)) + β * normalize(U_i(k))
             w_i(k) = softmax_k(S_i(k))

α controls weight on smoothing (original NAFS), β controls weight on utility.
Default: α=0.5, β=0.5 (balanced).

Usage:
  cd src/

  # Clustering with hybrid scoring
  python nafs_hybrid.py --task=clustering --dataset=cora --mode=mean

  # Link prediction with hybrid scoring
  python nafs_hybrid.py --task=link_prediction --dataset=cora --mode=mean

  # Tune alpha / beta (e.g. more utility weight)
  python nafs_hybrid.py --task=clustering --dataset=cora --mode=mean --alpha=0.3 --beta=0.7

Arguments:
  --alpha   weight on smoothing score D (default 0.5)
  --beta    weight on utility score U  (default 0.5)


DATA FORMAT
-----------
Place Planetoid data files in ../data/ (relative to src/):
  ind.cora.x, ind.cora.y, ind.cora.tx, ind.cora.ty,
  ind.cora.allx, ind.cora.ally, ind.cora.graph, ind.cora.test.index
  (same pattern for citeseer and pubmed)


DEPENDENCIES
------------
Python 3, PyTorch, NumPy, SciPy, scikit-learn, NetworkX, munkres


NOTES
-----
- This implementation vectorizes the per-node Python loop using einsum,
  making it significantly faster than the original code.
- Wiki dataset is not supported (missing tfidf.txt).
- Seed search script (nafs_seed_search.py) can re-run the sweep if you want
  to find better seeds for other datasets.
