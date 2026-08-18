"""Synthetic controlled experiments (Reviewer B: 'several synthetic
settings'). We generate graphs with KNOWN topological regimes and run the
exact same pipeline used on the co-click graph -- degree power-law fit,
1-skeleton Betti curves, and genuine flag-complex persistent homology
(beta_0, beta_1, beta_2 with expansion to tetrahedra) -- to show (a) the
pipeline distinguishes the regimes, and (b) which regime the real
news co-click graph matches.

Regimes:
  ER   Erdos-Renyi G(n,m)          -- no hub structure, no communities
  BA   Barabasi-Albert             -- power-law degree (scale-free)
  SBM  stochastic block model      -- planted communities
  WS   Watts-Strogatz small-world  -- high clustering, ring-like
"""
import json
import numpy as np
import networkx as nx
import gudhi
import warnings; warnings.filterwarnings('ignore')

rng = np.random.default_rng(0)
N = 1500


def powerlaw_gamma(degs):
    d = np.array([x for x in degs if x >= 5])
    if len(d) < 20:
        return None, None
    vals, cnt = np.unique(d, return_counts=True)
    p = cnt / cnt.sum()
    x, y = np.log(vals), np.log(p)
    a, b = np.polyfit(x, y, 1)
    ss_res = ((y - (a * x + b)) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return float(-a), float(r2)


def one_skeleton_betti(G):
    n, e = G.number_of_nodes(), G.number_of_edges()
    b0 = nx.number_connected_components(G)
    return b0, max(0, e - n + b0)


def flag_ph(G, max_dim=3):
    st = gudhi.SimplexTree()
    for v in G.nodes():
        st.insert([int(v)], 0.0)
    for u, v in G.edges():
        st.insert([int(u), int(v)], 1.0)
    st.expansion(max_dim)
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
    betti = list(st.betti_numbers()) + [0, 0, 0]
    return betti[0], betti[1], betti[2]


def analyse(name, G):
    G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
    degs = [d for _, d in G.degree()]
    gamma, r2 = powerlaw_gamma(degs)
    clustering = nx.average_clustering(G)
    b0_1, b1_1 = one_skeleton_betti(G)
    b0_f, b1_f, b2_f = flag_ph(G)
    row = {
        'graph': name, 'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(),
        'powerlaw_gamma': round(gamma, 2) if gamma else None,
        'powerlaw_r2': round(r2, 3) if r2 else None,
        'clustering': round(clustering, 3),
        'skeleton_beta0': b0_1, 'skeleton_beta1': b1_1,
        'flag_beta0': b0_f, 'flag_beta1': b1_f, 'flag_beta2': b2_f,
    }
    print(f"{name:24s} gamma={row['powerlaw_gamma']} R2={row['powerlaw_r2']} "
          f"C={row['clustering']:.3f} | skeleton(b0={b0_1},b1={b1_1}) "
          f"flag(b0={b0_f},b1={b1_f},b2={b2_f})")
    return row


def main():
    results = []
    # ER with ~ same average degree as a scale-free graph
    results.append(analyse('ER (random)', nx.gnm_random_graph(N, N * 4, seed=0)))
    # BA power-law / scale-free
    results.append(analyse('BA (scale-free)', nx.barabasi_albert_graph(N, 4, seed=0)))
    # SBM planted communities
    sizes = [N // 5] * 5
    P = np.full((5, 5), 0.002); np.fill_diagonal(P, 0.06)
    results.append(analyse('SBM (communities)', nx.stochastic_block_model(sizes, P, seed=0)))
    # WS small-world (high clustering)
    results.append(analyse('WS (small-world)', nx.watts_strogatz_graph(N, 8, 0.1, seed=0)))
    # a scale-free graph WITH triangles (BA + triangle formation) to mimic
    # the co-click regime: power-law AND high clustering
    results.append(analyse('Powerlaw+clustering (HK)',
                           nx.powerlaw_cluster_graph(N, 4, 0.5, seed=0)))
    json.dump(results, open('topology_data/synthetic_results.json', 'w'), indent=2)
    print('\nsaved topology_data/synthetic_results.json')


if __name__ == '__main__':
    main()
