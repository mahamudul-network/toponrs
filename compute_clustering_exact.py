"""Exact average local clustering coefficient over ALL nodes (not a
sample), used for the value reported in the graph-statistics table.
Uses triangle counts from A^2 restricted to the binary support.
"""
import argparse
import numpy as np
from scipy import sparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adj', default='topology_data/coclick_adj.npz')
    args = ap.parse_args()
    A = sparse.load_npz(args.adj).tocsr()
    Abin = (A > 0).astype(np.float64)
    Abin.setdiag(0); Abin.eliminate_zeros()
    deg = np.asarray(Abin.sum(axis=1)).ravel()
    tri = np.asarray((Abin @ Abin).multiply(Abin).sum(axis=1)).ravel() / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        cc = np.where(deg >= 2, 2 * tri / (deg * (deg - 1)), 0.0)
    print(f'exact average local clustering (all {len(deg)} nodes) = {cc.mean():.4f}')
    conn = deg > 0
    print(f'exact average local clustering (connected {int(conn.sum())} nodes) = {cc[conn].mean():.4f}')
    print(f'total triangles = {int(tri.sum() / 3)}')


if __name__ == '__main__':
    main()
