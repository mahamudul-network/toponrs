#!/usr/bin/env python3
"""Aggregate multi-seed runs and compute significance tests for the
TopoNRS revision (mean +/- std tables, paired tests vs the NRMS base).

Reads results/seed_runs/{variant}_seed{S}_results.json written by
run_all_experiments.py (which stores final metrics plus per-impression
metric arrays) and writes:
  results/seed_runs/aggregate_summary.json
  results/seed_runs/aggregate_table.txt   (LaTeX-ready rows)

Significance methodology:
  * across-seed: PAIRED t-test on the per-seed final metrics, aligned by
    the shared seed set (the two models use the same seeds);
  * per-impression paired: for each impression, the metric is averaged
    across seeds within each model, giving two paired vectors over the
    73,152 validation impressions; we report a paired t-test and a
    Wilcoxon signed-rank test on those vectors.
"""
import os, json, glob, argparse
from collections import defaultdict
import numpy as np
from scipy import stats

METRICS = ['auc', 'mrr', 'ndcg5', 'ndcg10']
LABELS = {'auc': 'AUC', 'mrr': 'MRR', 'ndcg5': 'nDCG@5', 'ndcg10': 'nDCG@10'}


def load_runs(run_dir):
    runs = defaultdict(list)  # variant -> list of result dicts
    for path in sorted(glob.glob(os.path.join(run_dir, '*_results.json'))):
        name = os.path.basename(path).replace('_results.json', '')
        if '_seed' not in name:
            continue
        variant = name.rsplit('_seed', 1)[0]
        with open(path) as f:
            data = json.load(f)
        if 'final' not in data:
            continue
        runs[variant].append(data)
    return runs


def per_impression_mean(runs, metric):
    """Average a per-impression metric across seeds, aligned by impression id."""
    acc = {}
    for r in runs:
        pi = r['per_impression']
        for iid, v in zip(pi['impression_ids'], pi[metric]):
            acc.setdefault(iid, []).append(v)
    ids = sorted(acc.keys())
    return ids, np.array([np.nanmean(acc[i]) for i in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='results/seed_runs')
    ap.add_argument('--treatment', default='e_best')
    ap.add_argument('--control', default='base')
    args = ap.parse_args()

    runs = load_runs(args.run_dir)
    if not runs:
        raise SystemExit(f"No *_results.json runs found in {args.run_dir}")

    summary = {}
    lines = []
    for variant in sorted(runs, key=lambda v: (v != 'base', v)):
        rs = runs[variant]
        seeds = [r['final']['seed'] for r in rs]
        row = {'n_seeds': len(rs), 'seeds': seeds}
        cells = []
        for m in METRICS:
            vals = np.array([r['final'][m] for r in rs])
            row[m] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                      'values': vals.tolist()}
            cells.append(f"{vals.mean():.4f} $\\pm$ {vals.std(ddof=1):.4f}" if len(vals) > 1
                         else f"{vals.mean():.4f}")
        summary[variant] = row
        lines.append(f"{variant:8s} (n={len(rs)}, seeds={seeds}) & " + " & ".join(cells) + " \\\\")

    # significance: treatment vs control
    sig = {}
    if args.treatment in runs and args.control in runs:
        # align by seed so the across-seed test is genuinely paired
        t_by_seed = {r['final']['seed']: r['final'] for r in runs[args.treatment]}
        c_by_seed = {r['final']['seed']: r['final'] for r in runs[args.control]}
        shared = sorted(set(t_by_seed) & set(c_by_seed))
        for m in METRICS:
            t_vals = np.array([t_by_seed[s][m] for s in shared])
            c_vals = np.array([c_by_seed[s][m] for s in shared])
            entry = {}
            if len(shared) > 1:
                # shared seeds -> PAIRED t-test (not Welch/independent)
                tt = stats.ttest_rel(t_vals, c_vals)
                entry['across_seed_paired_t'] = {'t': float(tt.statistic),
                                                 'p': float(tt.pvalue),
                                                 'n_seeds': len(shared)}
            ids_t, pi_t = per_impression_mean(runs[args.treatment], m)
            ids_c, pi_c = per_impression_mean(runs[args.control], m)
            assert ids_t == ids_c, "impression ids differ between models"
            ok = ~(np.isnan(pi_t) | np.isnan(pi_c))
            pt = stats.ttest_rel(pi_t[ok], pi_c[ok])
            entry['paired_t'] = {'t': float(pt.statistic), 'p': float(pt.pvalue),
                                 'n_impressions': int(ok.sum())}
            diff = pi_t[ok] - pi_c[ok]
            nz = diff != 0
            if nz.sum() > 0:
                w = stats.wilcoxon(diff[nz])
                entry['wilcoxon'] = {'W': float(w.statistic), 'p': float(w.pvalue)}
            entry['mean_improvement'] = float(np.nanmean(diff))
            sig[m] = entry

    out = {'per_variant': summary,
           'significance': {'treatment': args.treatment, 'control': args.control, 'tests': sig}}
    out_path = os.path.join(args.run_dir, 'aggregate_summary.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    table_path = os.path.join(args.run_dir, 'aggregate_table.txt')
    with open(table_path, 'w') as f:
        f.write("Variant & " + " & ".join(LABELS[m] for m in METRICS) + " \\\\\n")
        f.write("\n".join(lines) + "\n\n")
        for m, e in sig.items():
            f.write(f"{LABELS[m]}: {args.treatment} vs {args.control}: "
                    f"mean diff {e['mean_improvement']:+.4f}")
            if 'across_seed_paired_t' in e:
                f.write(f" | across-seed paired-t p={e['across_seed_paired_t']['p']:.4g}")
            f.write(f" | paired t p={e['paired_t']['p']:.4g}")
            if 'wilcoxon' in e:
                f.write(f" | Wilcoxon p={e['wilcoxon']['p']:.4g}")
            f.write("\n")

    print(open(table_path).read())
    print(f"Wrote {out_path} and {table_path}")


if __name__ == '__main__':
    main()
