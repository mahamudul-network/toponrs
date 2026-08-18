# Reproduction guide

Each paper table/figure below maps to one command and one output file.
Software: Python 3.10, PyTorch 2.8, Transformers 4.56, GUDHI 3.13,
NetworkX 3.4, SciPy 1.15. Seeds are fixed (42, 1, 2, 3, 4).

## Preprocessing and graphs
```
python preprocess_mind.py --train_dir ../mind_dataset/MINDsmall_train --dev_dir ../mind_dataset/MINDsmall_dev
python build_adjacency_only.py                         # -> topology_data/coclick_adj.npz
python build_coclick_sparse.py --behaviors ../mind_dataset/MINDlarge_train/behaviors.tsv --tag mindlarge
python build_neighbor_features.py                      # collaborative context (MIND-small)
python build_neighbor_features.py --adj topology_data/coclick_adj_mindlarge.npz \
       --news_ids topology_data/news_ids_mindlarge.json --out topology_data/neighbor_context_mindlarge.pkl \
       --train_dir ../mind_dataset/MINDlarge_train --dev_dir ../mind_dataset/MINDlarge_dev
```

## Topological analysis
| Paper item | Command | Output |
|---|---|---|
| Table 1 (Betti curves) & Fig. Betti | `python build_topology_graph.py` | `topology_data/betti_curves.json` (β₁ = \|E\|+β₀−N) |
| Fig. 1 (true degree) | `python make_degree_fig.py` | `figures/degree_distribution.pdf` (max degree 9,321) |
| Table 2 (flag-complex PH, H₂) | `python compute_persistent_homology.py --tag mindsmall --h2_thresh 40 --skip_skeleton` | `topology_data/ph_mindsmall.json` (β₁=β₂=0; expansion to tetrahedra, max_dim=3) |
| Table 2 (MIND-large row) | `python compute_persistent_homology.py --adj topology_data/coclick_adj_mindlarge.npz --tag mindlarge --h2_thresh 678 --skip_skeleton` | `topology_data/ph_mindlarge.json` |
| Table 3 (synthetic) | `python synthetic_experiments.py` | `topology_data/synthetic_results.json` |

## Model (MIND-small, 5 seeds; MIND-large, 3 seeds)
| Paper item | Command |
|---|---|
| Table 4 (main), Table 6 (ablation) | `bash run_seeds.sh 0 base 42 1 2 3 4` ; `bash run_seeds.sh 1 e_collab 42 1 2 3 4` ; `bash run_seeds.sh 0 e_best 42 1 2 3 4` ; `bash run_seeds.sh 1 h_best 42 1 2` |
| Table 7 (MIND-large) | `bash run_seeds_large.sh 0 base 42 1 2` ; `bash run_seeds_large.sh 1 e_collab 42 1 2` |
| Significance (Sec. 6.5) | `python aggregate_results.py --treatment e_best --control base` |

Variants: `base` = NRMS backbone (Euclidean dot product); `e_collab` =
TopoNRS with collaborative co-click context; `e_best` = collab + graph
statistics + 2-hop negatives; `h_best` = the same with the
Poincaré-distance scorer (reported as a negative result).
