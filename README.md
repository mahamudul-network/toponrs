# TopoNRS: Exploiting Co-Click Graph Topology for Neural News Recommendation via Hyperbolic Geometry

Code for the paper *"TopoNRS: Exploiting Co-Click Graph Topology for Neural
News Recommendation via Hyperbolic Geometry"* (International Journal of
Topology, under review).

The paper's primary contribution is a **topological analysis** of the
MIND-small co-click graph (degree distribution, Betti curves under a
weight filtration, Euler characteristic, and an Erdos-Renyi null-model
comparison). TopoNRS is a topology-informed model that tests whether the
exposed structure helps recommendation. Its mechanisms:

1. **Collaborative co-click context** (main): each article is represented
   partly by the co-click-weighted mean of its neighbours' title
   embeddings;
2. **Local co-click graph statistics** (degree, weight stats, clustering);
3. **Topology-guided hard negatives** (2-hop ambiguous neighbours);
4. **Optional Poincaré-ball scoring** motivated by the power-law degrees.

**Honest result:** across 5 seeds, TopoNRS matches the NRMS backbone in
mean accuracy (AUC 0.6759 vs 0.6745, paired p=0.66) while reducing seed
variance ~3x. We report this parity result, not an improvement.

## Setup

```bash
pip install -r requirements.txt
```

Download **MIND-small** from https://msnews.github.io/ and unzip into
`../mind_dataset/MINDsmall_train` and `../mind_dataset/MINDsmall_dev`
(or adjust the paths at the top of `run_all_experiments.py`).
We do not redistribute the dataset; see the Microsoft Research licence
terms on the MIND website.

## Pipeline

```bash
# 1. Preprocess raw MIND files (tokenised titles, negative-sampled behaviours)
python preprocess_mind.py --train_dir ../mind_dataset/MINDsmall_train \
                          --dev_dir   ../mind_dataset/MINDsmall_dev

# 2. Build the co-click graph, topology statistics, and Betti curves
#    (full analysis; ~1-2 h on CPU)
python build_topology_graph.py
#    ... or only the adjacency needed for training (~10 min):
python build_adjacency_only.py

# 2b. Build the co-click derived features used by the model
python build_2hop_negatives.py      # 2-hop ambiguous hard-negative candidates
python build_neighbor_features.py   # collaborative co-click context vectors

# 3. Train (single run)
python run_all_experiments.py --variant e_best --seed 42 --gpu 0
#    --variant base     = plain NRMS backbone (Euclidean dot-product scorer)
#    --variant e_collab = + collaborative co-click context (main mechanism)
#    --variant e_best   = collab + graph stats + 2-hop negatives (TopoNRS-Full)
#    --variant h_best   = e_best with the Poincaré-distance scorer
#    (base/gf/hyp/ns/full reproduce the original flawed design for the diagnosis)

# 4. Multi-seed runs (as reported in the paper; shared seeds for paired tests)
bash run_seeds.sh 0 base     42 1 2 3 4
bash run_seeds.sh 1 e_collab 42 1 2 3 4
bash run_seeds.sh 0 e_best   42 1 2 3 4
bash run_seeds.sh 1 h_best   42 1 2

# 5. Aggregate: mean +/- std tables and significance tests
#    (across-seed paired t-test + per-impression paired t-test / Wilcoxon)
python aggregate_results.py --treatment e_best --control base

# 6. Paper figures
python generate_paper_figures.py
```

Each run writes `results/seed_runs/{variant}_seed{S}_results.json`
containing the validation-metric trajectory, final metrics, and
per-impression AUC/MRR/nDCG arrays used by the significance tests.

## Hardware

One NVIDIA RTX 6000 Ada (48 GB) per run; a 6-epoch run on MIND-small
takes roughly 10 hours with validation every 500 steps. The BERT title
encoder is frozen, so memory requirements are modest (~3 GB).

## Citation

```bibtex
@article{hasan2026toponrs,
  title   = {TopoNRS: Exploiting Co-Click Graph Topology for Neural News
             Recommendation via Hyperbolic Geometry},
  author  = {Hasan, Mahamudul and Rahman, Anisur},
  journal = {International Journal of Topology},
  year    = {2026}
}
```
