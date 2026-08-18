"""Memory-efficient co-click graph builder via sparse M^T M, for large
datasets (MIND-large). Reads users' click histories, builds the binary
user x news incidence matrix M, forms the news x news co-click count
matrix M^T M, thresholds at tau, and saves the adjacency plus the same
topological statistics (degree distribution, components, 1-skeleton Betti
curves) used for MIND-small.
"""
import os, json, argparse
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from collections import Counter


def build(behaviors_path, tau=2):
    beh = pd.read_table(behaviors_path, header=None,
                        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    beh['clicked_news'] = beh['clicked_news'].fillna('')
    # user -> set of clicked (history) news
    user_ids, news_ids = {}, {}
    rows, cols = [], []
    for u, hist in zip(beh['user'], beh['clicked_news']):
        if not hist:
            continue
        ui = user_ids.setdefault(u, len(user_ids))
        for nid in set(hist.split()):
            ni = news_ids.setdefault(nid, len(news_ids))
            rows.append(ui); cols.append(ni)
    nU, nN = len(user_ids), len(news_ids)
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                          shape=(nU, nN))
    M.data[:] = 1
    print(f'  incidence: {nU} users x {nN} news, {M.nnz} clicks')
    C = (M.T @ M).tocsr()                 # news x news co-click counts
    C.setdiag(0); C.eliminate_zeros()
    C.data[C.data < tau] = 0; C.eliminate_zeros()
    print(f'  co-click edges (tau>={tau}): {C.nnz // 2}')
    idx2news = {v: k for k, v in news_ids.items()}
    news_list = [idx2news[i] for i in range(nN)]
    return C, news_list, {v: v for v in range(nN)}, nU


def stats_and_betti(C, nU):
    n = C.shape[0]
    deg = np.asarray((C > 0).sum(axis=1)).ravel()
    ncomp, labels = connected_components(C, directed=False)
    comp = Counter(labels); largest = max(comp.values())
    coo = sparse.triu(C, k=1).tocoo()
    w = coo.data.astype(float)
    stats = {
        'num_news_nodes': int(n), 'num_users': int(nU),
        'num_edges_coclick': int(C.nnz // 2),
        'max_degree': int(deg.max()), 'isolated_nodes': int((deg == 0).sum()),
        'isolated_frac': float((deg == 0).mean()),
        'num_connected_components': int(ncomp),
        'largest_component_size': int(largest),
        'largest_component_frac': float(largest / n),
    }
    # 1-skeleton Betti curves over a log-spaced weight filtration
    uw = np.unique(w)
    if len(uw) > 11:
        levels = np.unique(np.quantile(uw, np.linspace(0, 1, 11)).astype(int))
    else:
        levels = uw.astype(int)
    betti = []
    for eps in sorted(set(int(x) for x in levels), reverse=True):
        m = w >= eps
        rr, cc = coo.row[m], coo.col[m]
        if len(rr) == 0:
            continue
        sub = sparse.csr_matrix((np.ones(len(rr)*2),
              (np.concatenate([rr, cc]), np.concatenate([cc, rr]))), shape=(n, n))
        b0, _ = connected_components(sub, directed=False)
        nv = len(np.unique(np.concatenate([rr, cc])))
        b1 = int(len(rr) + b0 - n)
        betti.append({'threshold': int(eps), 'num_edges': int(m.sum()),
                      'num_vertices': int(nv), 'beta_0': int(b0),
                      'beta_1': max(0, b1)})
    return stats, betti, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--behaviors', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--tau', type=int, default=2)
    ap.add_argument('--out', default='./topology_data')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f'[{args.tag}] building co-click graph from {args.behaviors}')
    C, news_list, news2idx, nU = build(args.behaviors, args.tau)
    sparse.save_npz(os.path.join(args.out, f'coclick_adj_{args.tag}.npz'), C)
    json.dump(news_list, open(os.path.join(args.out, f'news_ids_{args.tag}.json'), 'w'))
    stats, betti, deg = stats_and_betti(C, nU)
    json.dump(stats, open(os.path.join(args.out, f'graph_statistics_{args.tag}.json'), 'w'), indent=2)
    json.dump(betti, open(os.path.join(args.out, f'betti_curves_{args.tag}.json'), 'w'), indent=2)
    ud, ct = np.unique(deg.astype(int), return_counts=True)
    json.dump(dict(zip(ud.tolist(), ct.tolist())),
              open(os.path.join(args.out, f'degree_distribution_{args.tag}.json'), 'w'))
    print(f'[{args.tag}] stats: {json.dumps(stats, indent=2)}')


if __name__ == '__main__':
    main()
