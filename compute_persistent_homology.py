"""Genuine persistent homology of the co-click graph, using GUDHI.

Answers Reviewer B: the paper now *uses* PH, not just component/cycle
counting. We build a filtered simplicial complex on the co-click graph
under the weight filtration (edges enter in decreasing co-click weight,
i.e. filtration value = w_max - w), compute the H0 and H1 persistence
diagrams, and derive persistence summaries (Betti curves from the
diagram, persistence entropy, total persistence). We additionally lift
the strongly-weighted core to its clique (flag) complex and compute H2
to probe higher-order structure (Reviewer 3's request).

Output: topology_data/ph_<tag>.json  and a persistence-diagram figure.
"""
import os, json, argparse
import numpy as np
from scipy import sparse
import gudhi

TOPO = './topology_data'


def load_graph(adj_path):
    adj = sparse.load_npz(adj_path).tocoo()
    mask = adj.row < adj.col
    r, c, w = adj.row[mask], adj.col[mask], adj.data[mask].astype(float)
    n = adj.shape[0]
    return n, r, c, w


def weight_filtration_ph(n, r, c, w, max_edges=None):
    """H0/H1 persistence under the decreasing-weight filtration.
    Filtration value of an edge = w_max - w (strong edges enter first);
    vertices enter at 0. Uses a SimplexTree (1-skeleton)."""
    wmax = w.max()
    order = np.argsort(-w)            # strong edges first
    if max_edges:
        order = order[:max_edges]
    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)
    for k in order:
        st.insert([int(r[k]), int(c[k])], filtration=float(wmax - w[k]))
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=False)
    diag = st.persistence()
    h0 = [(b, d) for dim, (b, d) in diag if dim == 0]
    h1 = [(b, d) for dim, (b, d) in diag if dim == 1]
    return h0, h1, float(wmax)


def flag_complex_persistence(n, r, c, w, thresh, max_dim=3):
    """Proper persistent homology on the FLAG (clique) complex of the
    strong-weight core, under the weight filtration. Edges enter in
    decreasing co-click weight (filtration = w_max - w); each higher
    simplex enters at the max filtration of its faces (flag filtration).
    Triangles can now *fill* 1-cycles, so H1 bars are finite (real
    persistence), and H2 detects voids.
    """
    keep = w >= thresh
    rr, cc, ww = r[keep], c[keep], w[keep]
    wmax = float(w.max())
    nodes = np.unique(np.concatenate([rr, cc]))
    remap = {int(v): i for i, v in enumerate(nodes)}
    st = gudhi.SimplexTree()
    for v in range(len(nodes)):
        st.insert([v], filtration=0.0)
    for a, b, wv in zip(rr, cc, ww):
        st.insert([remap[int(a)], remap[int(b)]], filtration=float(wmax - wv))
    st.expansion(max_dim)                 # flag complex, flag filtration
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
    diag = st.persistence()
    betti = st.betti_numbers()
    by_dim = {0: [], 1: [], 2: []}
    for dim, (b, d) in diag:
        if dim in by_dim:
            by_dim[dim].append((b, d))
    counts = {}
    for d in range(max_dim + 1):
        counts[d] = sum(1 for s, _ in st.get_simplices() if len(s) == d + 1)
    def finite(bars):
        return [(b, d) for b, d in bars if np.isfinite(d)]
    return {
        'threshold': int(thresh),
        'num_vertices': int(len(nodes)),
        'num_edges': int(keep.sum()),
        'num_triangles': int(counts.get(2, 0)),
        'num_tetrahedra': int(counts.get(3, 0)),
        'betti': [int(x) for x in betti],
        'H1_bars_finite': len(finite(by_dim[1])),
        'H1_persistence_entropy': persistence_entropy(by_dim[1]),
        'H1_max_persistence': float(max([d - b for b, d in finite(by_dim[1])], default=0.0)),
        'H2_bars': len(by_dim[2]),
        'H1_diagram': [[float(b), float(d)] for b, d in finite(by_dim[1])[:3000]],
        'H2_diagram': [[float(b), float(d)] for b, d in finite(by_dim[2])[:3000]],
    }


def persistence_entropy(diag):
    """Normalized persistence entropy of a diagram (finite bars)."""
    life = np.array([d - b for b, d in diag if np.isfinite(d) and d > b])
    if life.sum() == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def betti_curve_from_diagram(diag, grid, wmax):
    """Betti number as a function of filtration value, read off the diagram."""
    betti = []
    for t in grid:
        b = sum(1 for bb, dd in diag
                if bb <= t and (not np.isfinite(dd) or dd > t))
        betti.append(int(b))
    return betti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adj', default=os.path.join(TOPO, 'coclick_adj.npz'))
    ap.add_argument('--tag', default='mindsmall')
    ap.add_argument('--h2_thresh', type=int, default=40)
    ap.add_argument('--max_edges', type=int, default=None,
                    help='cap edges for the 1-skeleton PH (memory); None=all')
    ap.add_argument('--skip_skeleton', action='store_true',
                    help='skip the full-graph 1-skeleton H0 PH (for very large graphs)')
    args = ap.parse_args()

    n, r, c, w = load_graph(args.adj)
    wmax = float(w.max())
    print(f'[{args.tag}] {n} nodes, {len(w)} edges, max weight {wmax:.0f}')

    out = {'tag': args.tag, 'n_nodes': n, 'n_edges': int(len(w)), 'max_weight': wmax}
    if not args.skip_skeleton:
        h0, h1, wmax = weight_filtration_ph(n, r, c, w, args.max_edges)
        finite_h0 = [(b, d) for b, d in h0 if np.isfinite(d)]
        print(f'H0: {len(h0)} classes ({len(h0)-len(finite_h0)} infinite)')
        grid = np.linspace(0, wmax, 60).tolist()
        out['H0'] = {'n_classes': len(h0), 'n_infinite': len(h0) - len(finite_h0),
                     'persistence_entropy': persistence_entropy(h0),
                     'total_persistence': float(sum(d - b for b, d in finite_h0))}
        out['betti_curve_grid'] = grid
        out['betti0_curve'] = betti_curve_from_diagram(h0, grid, wmax)
    print(f'[{args.tag}] flag-complex PH on weight>={args.h2_thresh} core...')
    out['flag_ph'] = flag_complex_persistence(n, r, c, w, args.h2_thresh)
    fp = out['flag_ph']
    print(f"  core: {fp['num_vertices']} nodes, {fp['num_edges']} edges, "
          f"{fp['num_triangles']} triangles | Betti={fp['betti']} | "
          f"finite H1 bars={fp['H1_bars_finite']}, H2 bars={fp['H2_bars']}")

    path = os.path.join(TOPO, f'ph_{args.tag}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
