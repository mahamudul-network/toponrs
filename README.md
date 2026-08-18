# Persistent Homology of News Co-Click Graphs and a Negative Result for Topology-Informed Recommendation

Code for the paper of the same name (International Journal of Topology,
under review). The paper is primarily a **topological study** of the news
co-click graphs of MIND-small and MIND-large, and it reports an **honest
negative result** for recommendation: a topology-informed model
(collaborative co-click context) matches, but does not beat, a strong
content backbone on both datasets.

## What the code does

- **Co-click graph construction** from MIND click histories
  (`build_topology_graph.py`, and `build_coclick_sparse.py` for the large
  memory-efficient `M^T M` build).
- **Topological analysis**: true-degree distribution, weight-filtration
  Betti curves (1-skeleton), Erdős–Rényi null comparison.
- **Genuine persistent homology** of the flag (clique) complex up to H₂
  with GUDHI (`compute_persistent_homology.py`).
- **Synthetic controlled experiments** over known graph regimes
  (`synthetic_experiments.py`).
- **The TopoNRS model** (NRMS backbone + collaborative co-click context)
  and its multi-seed evaluation on MIND-small and MIND-large
  (`run_all_experiments.py`, `run_seeds.sh`, `run_seeds_large.sh`).

Note on terminology: throughout, **degree** means the number of co-click
*neighbours* (binary support of the adjacency), computed as `(adj>0).sum`.
The weighted row-sum (sum of shared-user counts) is a *different* quantity,
node **strength**, and is reported separately.

## Setup

```bash
pip install -r requirements.txt   # torch, transformers, scipy, gudhi, networkx, ...
```
Download **MIND-small** and **MIND-large** from https://msnews.github.io/
into `../mind_dataset/MINDsmall_{train,dev}` and
`../mind_dataset/MINDlarge_{train,dev}`. We do not redistribute the data.

See **REPRODUCE.md** for the exact command that produces each table/figure.
