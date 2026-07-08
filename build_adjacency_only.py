"""Rebuild only the co-click adjacency artifacts needed for training
(coclick_adj.npz, news2idx.json, news_ids.json), skipping the slow
statistics / persistence / hard-negative steps in build_topology_graph.py.
"""
import os
from scipy import sparse
import json

from build_topology_graph import build_user_news_graph, build_coclick_graph, MIND_TRAIN, OUTPUT_DIR


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_clicks, news_users = build_user_news_graph(
        os.path.join(MIND_TRAIN, 'behaviors.tsv'))
    adj, news_ids, news2idx = build_coclick_graph(news_users, min_shared_users=2)

    sparse.save_npz(os.path.join(OUTPUT_DIR, 'coclick_adj.npz'), adj)
    with open(os.path.join(OUTPUT_DIR, 'news_ids.json'), 'w') as f:
        json.dump(news_ids, f)
    with open(os.path.join(OUTPUT_DIR, 'news2idx.json'), 'w') as f:
        json.dump(news2idx, f)
    print(f"Saved adjacency ({adj.shape[0]} nodes, {adj.nnz // 2} edges) to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
