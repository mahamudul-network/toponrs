"""Rebuild only the 2-hop ambiguous hard-negative candidates
(structurally similar but NOT directly co-clicked) from the existing
co-click adjacency. These are the safe hard negatives: close in the
graph's 2-hop structure yet not direct co-clicks, so far less likely to
be false negatives than 1-hop co-click neighbours."""
import os, json, pickle
import numpy as np
from scipy import sparse
from tqdm import tqdm

TOPO = './topology_data'

def main():
    adj = sparse.load_npz(os.path.join(TOPO, 'coclick_adj.npz')).tocsr()
    news_ids = json.load(open(os.path.join(TOPO, 'news_ids.json')))
    n = adj.shape[0]
    adj_bin = (adj > 0).astype(np.float32)

    cand = {}
    for idx in tqdm(range(n), desc="2-hop negatives"):
        one_hop = set(adj_bin.indices[adj_bin.indptr[idx]:adj_bin.indptr[idx+1]])
        if not one_hop:
            cand[news_ids[idx]] = []
            continue
        # 2-hop reach = union of neighbours' neighbours
        two = adj_bin[list(one_hop)].sum(axis=0)
        two = np.asarray(two).ravel()
        two_hop = set(np.nonzero(two)[0]) - one_hop - {idx}
        if two_hop:
            ranked = sorted(two_hop, key=lambda c: -two[c])
            cand[news_ids[idx]] = [news_ids[c] for c in ranked[:20]]
        else:
            cand[news_ids[idx]] = []

    with open(os.path.join(TOPO, 'hard_neg_candidates.pkl'), 'wb') as f:
        pickle.dump(cand, f)
    has = sum(1 for v in cand.values() if v)
    print(f"{has}/{n} articles have 2-hop hard-negative candidates")


if __name__ == '__main__':
    main()
