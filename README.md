# Persistent Homology of News Co-Click Graphs and a Negative Result for Topology-Informed Recommendation

Code for the paper of the same name (*International Journal of Topology*,
manuscript `ijt-4339654`). The paper is primarily a **topological study** of the
news co-click graphs of MIND-small and MIND-large, and it reports an **honest
negative result** for recommendation: a topology-informed model (collaborative
co-click context) matches, but does not beat, a strong content backbone on both
datasets.

The revision tagged **`v1.1-accepted`** is the exact code corresponding to the
published article.

## What the code does

- **Co-click graph construction** from MIND click histories
  (`build_topology_graph.py`, and `build_coclick_sparse.py` for the large
  memory-efficient `Mᵀ M` build).
- **Topological analysis**: true-degree distribution, weight-filtration Betti
  curves (1-skeleton), Erdős–Rényi null comparison.
- **Genuine persistent homology** of the flag (clique) complex up to H₂ with
  GUDHI (`compute_persistent_homology.py`).
- **Synthetic controlled experiments** over known graph regimes
  (`synthetic_experiments.py`).
- **The TopoNRS model** (NRMS backbone + collaborative co-click context) and its
  multi-seed evaluation on MIND-small and MIND-large
  (`run_all_experiments.py`, `run_seeds.sh`, `run_seeds_large.sh`).
- **A verification harness** (`check_manuscript_numbers.py`) that re-derives every
  quantitative claim in the paper from the generated artefacts and the raw MIND
  files and reports PASS/FAIL per claim.

## Definitions used throughout

- **Degree** means the number of co-click *neighbours* (binary support of the
  adjacency), computed as `(adj > 0).sum`. The weighted row-sum (sum of
  shared-user counts) is a *different* quantity, node **strength**, and is
  reported separately. The whole-graph mean degree is `2|E|/N = 191.3` over all
  `N = 33,195` nodes; restricted to the `17,558` connected nodes it is `361.7`.
- The **co-click graph** is built from the *click-history field* of the
  **training** behaviours only. Clicked candidates inside the impression lists
  are not used, and no validation or test behaviour enters the construction. Its
  vertex set is the `33,195` articles that appear in at least one training click
  history, not the `51,282` articles listed in `MINDsmall_train/news.tsv`.
- The **flag-complex persistent homology** is computed on the strongly weighted
  cores where the clique expansion is tractable (e.g. `ε ≥ 40` on MIND-small:
  `951` nodes, `2.9%`, and `18,071` edges, `0.6%`). The `β₁ = β₂ = 0` finding is
  scoped to those cores; the full low-weight flag complex is out of reach.

## Setup

```bash
pip install -r requirements.txt   # torch, transformers, scipy, gudhi, networkx, ...
```

Download **MIND-small** and **MIND-large** from <https://msnews.github.io/> into
`../mind_dataset/MINDsmall_{train,dev}` and `../mind_dataset/MINDlarge_{train,dev}`.
We do not redistribute the data.

See **REPRODUCE.md** for the exact command that produces each table and figure.
