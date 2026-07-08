"""
Build Topology Graphs and Compute Topological Features from MIND Dataset.
Used for TopoNRS: Topology-Aware Neural Recommendation System.

This script:
1. Builds user-news bipartite graph from behaviors
2. Constructs news-news co-click graph (news items co-clicked by same users)
3. Computes graph topology statistics (degree, clustering, components)
4. Computes persistent homology features (Betti numbers, persistence diagrams)
5. Saves topology features for use in TopoNRS training
"""

import os
import sys
import json
import pickle
import time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

# Paths
MIND_TRAIN = '../mind_dataset/MINDsmall_train'
MIND_DEV = '../mind_dataset/MINDsmall_dev'
OUTPUT_DIR = './topology_data'


def build_user_news_graph(behaviors_path):
    """
    Build user-news bipartite graph from raw behaviors file.
    Returns:
        user_clicks: dict[user_id -> set of news_ids]
        news_users: dict[news_id -> set of user_ids]
    """
    print(f"Building user-news bipartite graph from {behaviors_path}")
    behaviors = pd.read_table(
        behaviors_path, header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions']
    )
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna('')

    user_clicks = defaultdict(set)
    news_users = defaultdict(set)

    for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), desc="Parsing clicks"):
        user = row['user']
        clicked = row['clicked_news'].split()
        for news_id in clicked:
            if news_id:
                user_clicks[user].add(news_id)
                news_users[news_id].add(user)

    print(f"  Users: {len(user_clicks)}, News: {len(news_users)}")
    print(f"  Total edges: {sum(len(v) for v in user_clicks.values())}")
    return user_clicks, news_users


def build_coclick_graph(news_users, min_shared_users=2):
    """
    Build news-news co-click graph: two news are connected if they share >= min_shared_users.
    Edge weight = number of shared users.
    """
    print(f"Building news-news co-click graph (min_shared_users={min_shared_users})")
    news_ids = sorted(news_users.keys())
    news2idx = {n: i for i, n in enumerate(news_ids)}
    n = len(news_ids)

    # Build adjacency via inverted index (user -> news)
    user_news = defaultdict(set)
    for news_id, users in news_users.items():
        for user in users:
            user_news[user].add(news_id)

    rows, cols, weights = [], [], []
    pair_counts = defaultdict(int)

    for user, news_set in tqdm(user_news.items(), desc="Computing co-clicks"):
        news_list = sorted(news_set)
        for i in range(len(news_list)):
            for j in range(i + 1, len(news_list)):
                pair_counts[(news_list[i], news_list[j])] += 1

    for (n1, n2), count in tqdm(pair_counts.items(), desc="Filtering edges"):
        if count >= min_shared_users:
            i, j = news2idx[n1], news2idx[n2]
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([count, count])

    adj = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    print(f"  Co-click graph: {n} nodes, {len(pair_counts)} unique pairs, "
          f"{adj.nnz // 2} edges (after filtering)")
    return adj, news_ids, news2idx


def compute_graph_statistics(adj, news_ids, user_clicks, news_users):
    """Compute graph topology statistics for the paper."""
    n = adj.shape[0]
    degrees = np.array(adj.sum(axis=1)).flatten()

    # Connected components
    n_components, labels = connected_components(adj, directed=False)

    # Largest component
    component_sizes = Counter(labels)
    largest_comp = max(component_sizes.values())

    # Degree statistics
    nonzero_degrees = degrees[degrees > 0]

    stats = {
        'num_news_nodes': n,
        'num_users': len(user_clicks),
        'num_edges_coclick': adj.nnz // 2,
        'avg_degree': float(np.mean(degrees)),
        'median_degree': float(np.median(degrees)),
        'max_degree': int(np.max(degrees)),
        'num_connected_components': n_components,
        'largest_component_size': largest_comp,
        'largest_component_fraction': largest_comp / n,
        'density': adj.nnz / (n * (n - 1)) if n > 1 else 0,
        'isolated_nodes': int(np.sum(degrees == 0)),
        'avg_clicks_per_user': float(np.mean([len(v) for v in user_clicks.values()])),
        'avg_users_per_news': float(np.mean([len(v) for v in news_users.values()])),
    }

    # Clustering coefficient (sample-based for efficiency)
    print("Computing clustering coefficients (sampled)...")
    sample_size = min(5000, n)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    clustering_coeffs = []
    adj_dense = None  # We'll use sparse operations

    for idx in tqdm(sample_indices, desc="Local clustering"):
        neighbors = adj[idx].nonzero()[1]
        k = len(neighbors)
        if k < 2:
            clustering_coeffs.append(0.0)
            continue
        # Count triangles
        subgraph = adj[neighbors][:, neighbors]
        triangles = subgraph.nnz / 2
        max_triangles = k * (k - 1) / 2
        clustering_coeffs.append(triangles / max_triangles)

    stats['avg_clustering_coefficient'] = float(np.mean(clustering_coeffs))
    stats['median_clustering_coefficient'] = float(np.median(clustering_coeffs))

    return stats


def compute_degree_distribution(adj):
    """Compute degree distribution for plotting."""
    degrees = np.array(adj.sum(axis=1)).flatten()
    unique_degrees, counts = np.unique(degrees.astype(int), return_counts=True)
    return dict(zip(unique_degrees.tolist(), counts.tolist()))


def compute_persistent_homology_features(adj, news_ids, news2idx, max_dim=1, max_filtration=10):
    """
    Compute persistent homology of the co-click graph using a Vietoris-Rips-like filtration
    based on weighted edges.

    We use a simple filtration: threshold the co-click weight from max to min.
    At each threshold, we compute Betti numbers.

    Returns:
        betti_curves: list of (threshold, beta_0, beta_1)
        news_ph_features: dict[news_id -> PH feature vector]
    """
    print("Computing persistent homology features...")

    # Get edge weights
    rows, cols = adj.nonzero()
    mask = rows < cols  # Upper triangle only
    rows, cols = rows[mask], cols[mask]
    weights = np.array(adj[rows, cols]).flatten()

    if len(weights) == 0:
        print("  Warning: No edges in graph, skipping PH")
        return [], {}

    # Filtration thresholds (from high weight to low weight)
    unique_weights = np.unique(weights)
    thresholds = sorted(unique_weights, reverse=True)
    if len(thresholds) > max_filtration:
        thresholds = thresholds[::len(thresholds) // max_filtration]

    n = adj.shape[0]
    betti_curves = []

    for thresh in tqdm(thresholds, desc="Filtration sweep"):
        # Build subgraph at this threshold
        mask_edges = weights >= thresh
        sub_rows = rows[mask_edges]
        sub_cols = cols[mask_edges]
        sub_adj = sparse.csr_matrix(
            (np.ones(len(sub_rows) * 2),
             (np.concatenate([sub_rows, sub_cols]),
              np.concatenate([sub_cols, sub_rows]))),
            shape=(n, n)
        )

        # Betti-0 = number of connected components
        n_components, _ = connected_components(sub_adj, directed=False)
        beta_0 = n_components

        # Betti-1 approximation: β₁ = |E| - |V| + β₀ (Euler characteristic)
        num_edges = len(sub_rows)
        num_vertices = len(np.unique(np.concatenate([sub_rows, sub_cols])))
        beta_1 = max(0, num_edges - num_vertices + n_components)

        betti_curves.append({
            'threshold': float(thresh),
            'beta_0': int(beta_0),
            'beta_1': int(beta_1),
            'num_edges': int(num_edges),
            'num_vertices': int(num_vertices)
        })

    # Per-news PH features: local Betti numbers at different scales
    print("Computing per-news topological features...")
    news_ph_features = {}
    for idx in tqdm(range(n), desc="News PH features"):
        neighbors = adj[idx].nonzero()[1]
        k = len(neighbors)
        if k == 0:
            # Isolated node
            news_ph_features[news_ids[idx]] = np.zeros(6, dtype=np.float32)
            continue

        neighbor_weights = np.array(adj[idx, neighbors].todense()).flatten()
        # Local features
        feat = np.array([
            k,                           # degree (local β₀ proxy)
            np.mean(neighbor_weights),   # avg edge weight
            np.std(neighbor_weights),    # weight variability
            np.max(neighbor_weights),    # strongest connection
            np.min(neighbor_weights),    # weakest connection
            0.0                          # local clustering (computed below)
        ], dtype=np.float32)

        # Local clustering coefficient
        if k >= 2:
            subgraph = adj[neighbors][:, neighbors]
            triangles = subgraph.nnz / 2
            feat[5] = triangles / (k * (k - 1) / 2)

        news_ph_features[news_ids[idx]] = feat

    return betti_curves, news_ph_features


def build_hard_negative_candidates(adj, news_ids, news2idx, top_k=20):
    """
    For each news item, find topology-guided hard negative candidates.
    These are 2-hop neighbors (structurally close but not directly co-clicked).

    The idea: if two news items share many graph neighbors but are NOT directly
    connected, they are "structurally similar but behaviorally different" —
    making them ideal hard negatives.
    """
    print("Building topology-guided hard negative candidates...")
    n = adj.shape[0]

    # 2-hop adjacency (adj^2 minus adj and diagonal)
    adj_binary = (adj > 0).astype(float)
    two_hop = adj_binary @ adj_binary

    hard_neg_candidates = {}
    for idx in tqdm(range(n), desc="Hard negatives"):
        # 2-hop neighbors not in 1-hop
        two_hop_neighbors = set(two_hop[idx].nonzero()[1])
        one_hop_neighbors = set(adj_binary[idx].nonzero()[1])
        candidates = two_hop_neighbors - one_hop_neighbors - {idx}

        if candidates:
            # Rank by 2-hop connection strength
            cand_list = list(candidates)
            cand_scores = [two_hop[idx, c] for c in cand_list]
            ranked = sorted(zip(cand_list, cand_scores), key=lambda x: -x[1])
            hard_neg_candidates[news_ids[idx]] = [
                news_ids[r[0]] for r in ranked[:top_k]
            ]
        else:
            hard_neg_candidates[news_ids[idx]] = []

    total_candidates = sum(len(v) for v in hard_neg_candidates.values())
    has_candidates = sum(1 for v in hard_neg_candidates.values() if v)
    print(f"  {has_candidates}/{n} news have hard negative candidates")
    print(f"  Average candidates per news: {total_candidates / max(1, has_candidates):.1f}")
    return hard_neg_candidates


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    start = time.time()

    # Step 1: Build user-news bipartite graph
    user_clicks, news_users = build_user_news_graph(
        os.path.join(MIND_TRAIN, 'behaviors.tsv'))

    # Step 2: Build co-click graph
    adj, news_ids, news2idx = build_coclick_graph(news_users, min_shared_users=2)

    # Step 3: Graph statistics
    stats = compute_graph_statistics(adj, news_ids, user_clicks, news_users)
    print("\n" + "=" * 60)
    print("GRAPH TOPOLOGY STATISTICS")
    print("=" * 60)
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print("=" * 60)

    # Save stats
    with open(os.path.join(OUTPUT_DIR, 'graph_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Step 4: Degree distribution
    degree_dist = compute_degree_distribution(adj)
    with open(os.path.join(OUTPUT_DIR, 'degree_distribution.json'), 'w') as f:
        json.dump(degree_dist, f, indent=2)

    # Step 5: Persistent homology
    betti_curves, news_ph_features = compute_persistent_homology_features(
        adj, news_ids, news2idx)

    with open(os.path.join(OUTPUT_DIR, 'betti_curves.json'), 'w') as f:
        json.dump(betti_curves, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'news_ph_features.pkl'), 'wb') as f:
        pickle.dump(news_ph_features, f)

    # Step 6: Hard negative candidates
    hard_neg_candidates = build_hard_negative_candidates(
        adj, news_ids, news2idx, top_k=20)

    with open(os.path.join(OUTPUT_DIR, 'hard_neg_candidates.pkl'), 'wb') as f:
        pickle.dump(hard_neg_candidates, f)

    # Step 7: Save adjacency matrix and mappings
    sparse.save_npz(os.path.join(OUTPUT_DIR, 'coclick_adj.npz'), adj)
    with open(os.path.join(OUTPUT_DIR, 'news_ids.json'), 'w') as f:
        json.dump(news_ids, f)
    with open(os.path.join(OUTPUT_DIR, 'news2idx.json'), 'w') as f:
        json.dump(news2idx, f)

    elapsed = time.time() - start
    print(f"\nTopology graph building completed in {elapsed:.1f}s")
    print(f"Output saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
