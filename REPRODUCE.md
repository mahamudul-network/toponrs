# Reproduction guide

Each table/figure of the paper below maps to one command and one output file.
Table and figure numbers refer to the **published version**.

Software: Python 3.10, PyTorch 2.8, Transformers 4.56, GUDHI 3.13,
NetworkX 3.4, SciPy 1.15, matplotlib 3.8.
Seeds are fixed: 42, 1, 2, 3, 4 (MIND-small) and 42, 1, 2 (MIND-large).

Download MIND-small and MIND-large from <https://msnews.github.io/> into
`../mind_dataset/MINDsmall_{train,dev}` and `../mind_dataset/MINDlarge_{train,dev}`.
We do not redistribute the data. All commands are run from the repository root.

## 1. Preprocessing and graph construction

```bash
python preprocess_mind.py --train_dir ../mind_dataset/MINDsmall_train \
                          --dev_dir   ../mind_dataset/MINDsmall_dev
python build_adjacency_only.py                 # -> topology_data/coclick_adj.npz
python build_coclick_sparse.py --behaviors ../mind_dataset/MINDlarge_train/behaviors.tsv \
                               --tag mindlarge # -> topology_data/coclick_adj_mindlarge.npz
python build_2hop_negatives.py                 # -> topology_data/hard_neg_candidates.pkl
python build_neighbor_features.py              # -> topology_data/neighbor_context.pkl
python build_neighbor_features.py \
       --adj      topology_data/coclick_adj_mindlarge.npz \
       --news_ids topology_data/news_ids_mindlarge.json \
       --out      topology_data/neighbor_context_mindlarge.pkl \
       --train_dir ../mind_dataset/MINDlarge_train \
       --dev_dir   ../mind_dataset/MINDlarge_dev
```

**Scope of the graph (Section 3 of the paper).** The co-click graph is built
from the **click-history field** of the *training* `behaviors.tsv` only. Clicked
candidates inside the impression lists are not used, and no validation or test
behaviour enters the construction. Its vertex set is therefore the
**33,195** articles that appear in at least one training click history — not the
51,282 articles listed in `MINDsmall_train/news.tsv`. `build_neighbor_features.py`
reads the validation split for article *titles* only (never behaviours), and every
article missing from the graph receives the zero vector.

## 2. Topological analysis

| Paper item | Command | Output |
|---|---|---|
| Table 1 (filtration levels) and Figure 2 (Betti curves) | `python build_topology_graph.py` | `topology_data/betti_curves.json` (β₁ = \|E\| + β₀ − N, with N = 33,195 held fixed across the filtration) |
| Table 4 (graph statistics), exact clustering 0.371 | `python compute_clustering_exact.py` | stdout |
| Figure 1 (true degree distribution) | `python make_degree_fig.py` | `figures/degree_distribution.pdf` and `submission/figures/` (max degree 9321; whole-graph mean 191.3; connected-node mean 361.7) |
| Table 2, MIND-small rows (flag-complex PH) | `python compute_persistent_homology.py --tag mindsmall --h2_thresh 40  --skip_skeleton`<br>`python compute_persistent_homology.py --tag mindsmall --h2_thresh 112 --skip_skeleton` | `topology_data/ph_mindsmall.json` (β₁ = β₂ = 0; flag expansion to tetrahedra, `max_dim=3`) |
| Table 2, MIND-large rows | `python compute_persistent_homology.py --adj topology_data/coclick_adj_mindlarge.npz --tag mindlarge --h2_thresh 678  --skip_skeleton`<br>`... --h2_thresh 1000 --skip_skeleton` | `topology_data/ph_mindlarge.json` |
| Table 3 (synthetic controls) | `python synthetic_experiments.py` | `topology_data/synthetic_results.json` |
| Figures 2, 4 (Betti curves, feature distributions) | `python generate_paper_figures.py` | `figures/*.pdf`, `submission/figures/*.pdf` |
| Figure 5 (training dynamics) | `python make_convergence_fig.py` | `figures/training_convergence.pdf` |

> **Note.** Each `compute_persistent_homology.py` run overwrites
> `topology_data/ph_<tag>.json`, so copy the file (e.g. to
> `ph_mindsmall_eps112.json`) between the two threshold runs if you want to keep
> both. The reported cores are small by construction — the MIND-small
> ε ≥ 40 core is 951 of 33,195 nodes (2.9%) and 18,071 of 3,175,187 edges (0.6%) —
> and the paper scopes the β₁ = β₂ = 0 finding to them.

> **Note on Table 3.** The `−s (R²)` column is a *descriptive* log–log tail slope,
> not a validated power-law exponent. `synthetic_experiments.py` fits degrees ≥ 5;
> `make_degree_fig.py` fits the MIND degrees ≥ 10. The MIND row of Table 3 is
> reported on the ε ≥ 40 core so that its skeleton and flag Betti numbers refer to
> one and the same graph.

## 3. Model training and evaluation

| Paper item | Command |
|---|---|
| Table 6 (main results) and Table 7 (ablation) | `bash run_seeds.sh 0 base       42 1 2 3 4`<br>`bash run_seeds.sh 1 e_collab   42 1 2 3 4`<br>`bash run_seeds.sh 0 e_best     42 1 2 3 4`<br>`bash run_seeds.sh 1 e_best_hyp 42 1 2 3 4` |
| Table 8 (MIND-large) | `bash run_seeds_large.sh 0 base 42 1 2`<br>`bash run_seeds_large.sh 1 e_collab 42 1 2` |
| Significance tests (Section 6.5, 7.1, 7.3) | `python aggregate_results.py --treatment e_best --control base` |

Variants: `base` = NRMS backbone (Euclidean dot product); `e_collab` = TopoNRS
with the collaborative co-click context; `e_best` = collab + graph statistics +
2-hop negatives (the Euclidean Full model); `e_best_hyp` = **identical to
`e_best` except for the Poincaré-distance scorer and the radial-spread
projection it requires** (the controlled scorer ablation, reported as a negative
result). `h_best`, `h_topo` and `full` are earlier, differently configured
variants kept for the record; they are **not** reported in the paper — use
`e_best_hyp` for the scorer ablation. **No reported model uses the optional
topology regulariser** (`reg=False`, λ = 0 throughout).

## 4. Evaluation protocol (stated for transparency)

MIND withholds the test labels. Following common practice on this benchmark,
`train_model()` both selects the checkpoint by validation AUC (evaluated every
500 optimisation steps) and reports the final metrics on the same official
validation split. This makes the absolute numbers mildly optimistic, but the
protocol is identical for every model and seed, so the paired comparisons the
paper's conclusions rest on are unaffected. See the Limitations section of the
paper.

## 5. Checking the manuscript numbers

`python check_manuscript_numbers.py` re-derives every quantitative claim in the
paper from the generated artefacts and the raw MIND files, and reports PASS/FAIL
per claim.
