#!/usr/bin/env python3
"""Re-derive every quantitative claim in the manuscript from the canonical
artefacts and report PASS/FAIL for each.

    python check_manuscript_numbers.py [--tex PROOF_FINAL/ijt-4339654.tex]
                                       [--mind ../mind_dataset] [--skip-raw]

Sources of truth, in order of preference:
  raw MIND files        -> dataset counts
  topology_data/*.json  -> graph, Betti, flag-PH and synthetic values
  results/seed_runs/*   -> per-seed metrics and significance tests

Exit code 0 iff every check passes.
"""
import argparse, glob, io, json, os, re, sys
from collections import defaultdict

import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components

OK, BAD = [], []


def check(name, got, want, tol=0.0, unit=''):
    """Record a numeric comparison."""
    if want is None or got is None:
        BAD.append((name, got, want, 'missing input')); return
    ok = abs(float(got) - float(want)) <= tol
    (OK if ok else BAD).append((name, got, want, unit))


def in_tex(name, tex, *needles):
    """Record that each literal claim string occurs in the manuscript."""
    for nd in needles:
        flat = re.sub(r'\s+', ' ', tex)
        (OK if nd in flat else BAD).append((f'{name}: "{nd}"', 'present' if nd in flat
                                            else 'ABSENT', 'present', ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tex', default='PROOF_FINAL/ijt-4339654.tex')
    ap.add_argument('--mind', default='../mind_dataset')
    ap.add_argument('--topo', default='topology_data')
    ap.add_argument('--runs', default='results/seed_runs')
    ap.add_argument('--skip-raw', action='store_true',
                    help='skip the (slow) recount of the raw MIND behaviour files')
    a = ap.parse_args()
    tex = io.open(a.tex, encoding='utf-8').read() if os.path.exists(a.tex) else ''
    if not tex:
        print(f'! manuscript not found at {a.tex}; skipping text checks')

    # ---------------------------------------------------------------- dataset
    if not a.skip_raw and os.path.isdir(a.mind):
        tr = os.path.join(a.mind, 'MINDsmall_train')
        dv = os.path.join(a.mind, 'MINDsmall_dev')
        hist, news_hist = defaultdict(set), set()
        n_imp = 0
        with io.open(os.path.join(tr, 'behaviors.tsv'), encoding='utf-8') as f:
            for line in f:
                p = line.rstrip('\n').split('\t'); n_imp += 1
                for nid in p[3].split():
                    hist[p[1]].add(nid); news_hist.add(nid)
        users_all = set()
        with io.open(os.path.join(tr, 'behaviors.tsv'), encoding='utf-8') as f:
            for line in f: users_all.add(line.split('\t')[1])
        n_dev = sum(1 for _ in io.open(os.path.join(dv, 'behaviors.tsv'), encoding='utf-8'))
        ntr = sum(1 for _ in io.open(os.path.join(tr, 'news.tsv'), encoding='utf-8'))
        ids = set()
        for d in (tr, dv):
            for line in io.open(os.path.join(d, 'news.tsv'), encoding='utf-8'):
                ids.add(line.split('\t', 1)[0])
        check('train impressions', n_imp, 156965)
        check('train users', len(users_all), 50000)
        check('train news.tsv articles', ntr, 51282)
        check('dev impressions', n_dev, 73152)
        check('train+dev news union', len(ids), 65238)
        check('articles in >=1 training history (graph nodes)', len(news_hist), 33195)
        check('users with non-empty history', len(hist), 49108)
        check('users with >=2 distinct history articles',
              sum(1 for v in hist.values() if len(v) >= 2), 48295)
        if tex:
            in_tex('dataset sentence', tex, '$156{,}965$ impressions from $50{,}000$ users',
                   '$51{,}282$ articles', '$48{,}295$ of them list at least two distinct')
            for gone in ('94{,}057', '65{,}238'):
                (BAD if gone in re.sub(r'\s+', ' ', tex) else OK).append(
                    (f'stale dataset number {gone} removed',
                     'still present' if gone in tex else 'gone', 'gone', ''))

    # ------------------------------------------------------- graph statistics
    gs = json.load(open(os.path.join(a.topo, 'graph_statistics.json')))
    N, E = gs['num_news_nodes'], gs['num_edges_coclick']
    check('graph nodes', N, 33195)
    check('graph edges', E, 3175187)
    check('isolated nodes', gs['isolated_nodes'], 15637)
    check('isolated fraction (%)', 100 * gs['isolated_nodes'] / N, 47.1, 0.05)
    check('connected components', gs['num_connected_components'], 15652)
    check('largest component', gs['largest_component_size'], 17529)
    check('largest component (%)', 100 * gs['largest_component_size'] / N, 52.8, 0.05)
    check('max weighted strength', gs['max_degree'], 146213)
    check('avg clicks per user', gs['avg_clicks_per_user'], 18.63, 0.005)
    check('avg users per article', gs['avg_users_per_article']
          if 'avg_users_per_article' in gs else gs['avg_users_per_news'], 27.56, 0.005)

    dt = json.load(open(os.path.join(a.topo, 'degree_true_mindsmall.json')))
    deg, cnt = np.array(dt['degree']), np.array(dt['count'])
    n_conn, sum_deg = int(cnt.sum()), int((deg * cnt).sum())
    check('sum of degrees == 2|E|', sum_deg, 2 * E)
    check('connected nodes', n_conn, 17558)
    check('max degree', dt['max_degree'], 9321)
    check('whole-graph mean degree 2|E|/N', sum_deg / N, 191.3, 0.05)
    check('connected-node mean degree', sum_deg / n_conn, 361.7, 0.05)
    med = deg[np.searchsorted(np.cumsum(cnt), n_conn / 2)]
    check('median connected degree', med, 52)
    check('max degree / whole-graph mean', dt['max_degree'] / (sum_deg / N), 48.7, 0.1)
    check('max degree / connected mean', dt['max_degree'] / (sum_deg / n_conn), 25.8, 0.1)

    p_er = 2 * E / (N * (N - 1)); thr_er = np.log(N) / N
    check('ER edge probability (x1e-3)', 1e3 * p_er, 5.8, 0.05)
    check('ER connectivity threshold ln n/n (x1e-4)', 1e4 * thr_er, 3.1, 0.05)
    check('ER p / threshold ratio', p_er / thr_er, 18.4, 0.2)
    check('ER expected clustering 2m/n^2 (x1e-3)', 1e3 * 2 * E / N**2, 5.8, 0.05)
    if tex:
        in_tex('mean-degree wording', tex,
               'whole-graph mean degree is $2|\\mathcal{E}|/|\\mathcal{N}| = 191.3$',
               'mean degree is $361.7$')
        in_tex('ER comparison', tex, 'about $18$ times the')
        in_tex('AUC tie term inside the sum', tex,
               '\\mathbb{1}\\!\\left[\\hat{y}_{ip} = \\hat{y}_{in}\\right] \\right),')
        for gone in ('fixed before the results were examined',
                     'ablation---are unaffected',
                     'TopoNRS reduces to standard NRMS training',
                     'significantly \\emph{lowers}',
                     'Replication on a Second Dataset',
                     'generalises across'):
            (BAD if gone in re.sub(r'\s+', ' ', tex) else OK).append(
                (f'retracted phrasing removed: "{gone}"',
                 'still present' if gone in tex else 'gone', 'gone', ''))

    # ------------------------------------------------ Betti curves (Table 1)
    bt = json.load(open(os.path.join(a.topo, 'betti_curves.json')))
    for r in bt:
        b1 = r['num_edges'] + r['beta_0'] - N
        check(f"Euler identity at eps={int(r['threshold'])}", r['beta_1'], max(0, b1))
    row40 = [r for r in bt if int(r['threshold']) == 40][0]
    row4 = [r for r in bt if int(r['threshold']) == 4][0]
    check('beta_1 at eps=40', row40['beta_1'], 17121)
    check('beta_1 at eps=4', row4['beta_1'], 1009368)
    check('chi at eps=4', row4['beta_0'] - row4['beta_1'], -985592)
    check('beta_0 drop across filtration (%)',
          100 * (33194 - row4['beta_0']) / 33194, 28.4, 0.5)

    # --------------------------------------------- flag-complex PH (Table 2)
    fs = {int(d['threshold']): d for d in
          json.load(open(os.path.join(a.topo, 'flag_betti_scan_mindsmall.json')))}
    fl = {int(d['threshold']): d for d in
          json.load(open(os.path.join(a.topo, 'flag_betti_scan_mindlarge.json')))}
    for tag, tbl, tot_n, tot_e, rows in (
            ('small', fs, N, E, [(40, 951, 18071, 387365, 6818828),
                                 (112, 235, 1828, 12257, 66838)]),
            ('large', fl, 79546, 26678885, [(678, 739, 12510, 229518, 3484272),
                                            (1000, 442, 5524, 66137, 652197)])):
        for eps, v, e, tri, tet in rows:
            d = tbl[eps]
            check(f'{tag} eps>={eps} nodes', d['V'], v)
            check(f'{tag} eps>={eps} edges', d['E'], e)
            check(f'{tag} eps>={eps} triangles', d['triangles'], tri)
            check(f'{tag} eps>={eps} tetrahedra', d['tetrahedra'], tet)
            check(f'{tag} eps>={eps} beta_1', d['betti1'], 0)
            check(f'{tag} eps>={eps} beta_2', d['betti2'], 0)
    # Table 3 MIND rows: full graph and eps>=40 core must each be internally
    # consistent, and the core statistics must NOT be the full-graph ones.
    check('Table 3 full-graph beta_1 (1-skeleton, tau=2)',
          E + gs['num_connected_components'] - N, 3157644)
    A = sparse.load_npz(os.path.join(a.topo, 'coclick_adj.npz')).tocsr()
    rr, cc = A.nonzero(); mm = rr < cc; rr, cc = rr[mm], cc[mm]
    ww = np.asarray(A[rr, cc]).ravel()
    kp = ww >= 40
    nd = np.unique(np.concatenate([rr[kp], cc[kp]]))
    rmp = {int(v): i for i, v in enumerate(nd)}
    ri = np.array([rmp[int(x)] for x in rr[kp]]); ci = np.array([rmp[int(x)] for x in cc[kp]])
    S = sparse.csr_matrix((np.ones(len(ri) * 2),
                           (np.concatenate([ri, ci]), np.concatenate([ci, ri]))),
                          shape=(len(nd), len(nd)))
    S.data[:] = 1.0; S.setdiag(0); S.eliminate_zeros()
    dg = np.asarray(S.sum(1)).ravel()
    tr = np.asarray((S @ S).multiply(S).sum(1)).ravel() / 2.0
    cl = np.where(dg >= 2, 2 * tr / (dg * (dg - 1)), 0.0)
    check('Table 3 core clustering', cl.mean(), 0.778, 0.001)
    uq, ct = np.unique(dg[dg > 0].astype(int), return_counts=True)
    pk = ct / ct.sum(); sl = uq >= 10
    sa, sb = np.polyfit(np.log(uq[sl]), np.log(pk[sl]), 1)
    sr2 = 1 - ((np.log(pk[sl]) - (sa * np.log(uq[sl]) + sb)) ** 2).sum() / \
              ((np.log(pk[sl]) - np.log(pk[sl]).mean()) ** 2).sum()
    check('Table 3 core log-log slope', sa, -0.68, 0.005)
    check('Table 3 core slope R2', sr2, 0.66, 0.005)
    b0c, _ = connected_components(S, directed=False)
    check('eps>=40 core is connected', b0c, 1)
    check('cycle claim: edges on a cycle (%)',
          100 * (int(kp.sum()) - (len(nd) - 1)) / int(kp.sum()), 94.7, 0.05)
    check('MIND-small eps>=40 node coverage (%)', 100 * 951 / N, 2.9, 0.05)
    check('MIND-small eps>=40 edge coverage (%)', 100 * 18071 / E, 0.6, 0.05)
    check('MIND-large eps>=678 node coverage (%)', 100 * 739 / 79546, 0.9, 0.05)
    check('MIND-large eps>=678 edge coverage (%)', 100 * 12510 / 26678885, 0.05, 0.005)
    ph = json.load(open(os.path.join(a.topo, 'ph_mindsmall.json')))
    check('finite H1 bars on MIND-small core', ph['flag_ph']['H1_bars_finite'], 3)

    # ------------------------------------------------ MIND-large graph stats
    gl = json.load(open(os.path.join(a.topo, 'graph_statistics_mindlarge.json')))
    check('MIND-large articles', gl['num_news_nodes'], 79546)
    check('MIND-large users', gl['num_users'], 698365)
    check('MIND-large edges', gl['num_edges_coclick'], 26678885)
    check('MIND-large components', gl['num_connected_components'], 25207)
    check('MIND-large isolated', gl['isolated_nodes'], 25198)
    check('MIND-large isolated (%)', 100 * gl['isolated_frac'], 31.7, 0.05)
    check('MIND-large largest component', gl['largest_component_size'], 54332)
    check('MIND-large largest (%)', 100 * gl['largest_component_frac'], 68.3, 0.05)
    check('MIND-large small-component nodes',
          gl['num_news_nodes'] - gl['isolated_nodes'] - gl['largest_component_size'], 16)
    check('MIND-large non-isolated components',
          gl['num_connected_components'] - gl['isolated_nodes'], 9)

    # ----------------------------------------------- synthetic table (Tab. 3)
    syn = {d['graph']: d for d in
           json.load(open(os.path.join(a.topo, 'synthetic_results.json')))}
    for g, gam, r2, C, b1s, b1f, b2f in (
            ('ER (random)', 3.58, 0.706, 0.006, 4502, 4407, 0),
            ('BA (scale-free)', 1.84, 0.805, 0.027, 4485, 3918, 2),
            ('SBM (communities)', 1.29, 0.105, 0.047, 13903, 9022, 1),
            ('WS (small-world)', None, 0.069, 0.476, 4501, 600, 15),
            ('Powerlaw+clustering (HK)', 1.67, 0.761, 0.235, 4477, 1808, 1)):
        d = syn[g]
        if gam is not None:
            check(f'{g} tail slope', d['powerlaw_gamma'], gam, 0.005)
        check(f'{g} R2', d['powerlaw_r2'], r2, 0.005)
        check(f'{g} clustering', d['clustering'], C, 0.0005)
        check(f'{g} skeleton beta_1', d['skeleton_beta1'], b1s)
        check(f'{g} flag beta_1', d['flag_beta1'], b1f)
        check(f'{g} flag beta_2', d['flag_beta2'], b2f)

    # ----------------------------------------- model results and significance
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.runs, '*_results.json'))):
        nm = os.path.basename(p).replace('_results.json', '')
        if '_seed' not in nm:
            continue
        v, s = nm.rsplit('_seed', 1)
        d = json.load(open(p))
        if 'final' in d:
            runs.setdefault(v, {})[int(s)] = d
    M = ['auc', 'mrr', 'ndcg5', 'ndcg10']
    want = {'base':           (0.6745, 0.0074, 0.3174, 0.0073, 0.3528, 0.0087, 0.4153, 0.0074),
            'e_collab':       (0.6765, 0.0038, 0.3180, 0.0045, 0.3527, 0.0050, 0.4160, 0.0046),
            'e_best':         (0.6759, 0.0023, 0.3194, 0.0027, 0.3534, 0.0032, 0.4166, 0.0027),
            'e_best_hyp':     (0.6699, 0.0036, 0.3136, 0.0043, 0.3448, 0.0059, 0.4098, 0.0046),
            'base_large':     (0.6907, 0.0016, 0.3314, 0.0032, 0.3675, 0.0033, 0.4310, 0.0022),
            'e_collab_large': (0.6903, 0.0021, 0.3311, 0.0025, 0.3662, 0.0036, 0.4309, 0.0027)}
    for v, w in want.items():
        seeds = sorted(runs[v])
        check(f'{v}: number of seeds', len(seeds), 5 if not v.endswith('large') else 3)
        for i, m in enumerate(M):
            x = np.array([runs[v][s]['final'][m] for s in seeds])
            check(f'{v} {m} mean', x.mean(), w[2 * i], 5e-5)
            check(f'{v} {m} std', x.std(ddof=1), w[2 * i + 1], 5e-5)
    # per-seed AUC lists quoted under Table 7, in seed order 42, 1, 2, 3, 4
    quoted = {'base':       [0.6637, 0.6702, 0.6809, 0.6781, 0.6798],
              'e_collab':   [0.6712, 0.6772, 0.6814, 0.6778, 0.6748],
              'e_best':     [0.6734, 0.6767, 0.6741, 0.6763, 0.6793],
              'e_best_hyp': [0.6677, 0.6715, 0.6658, 0.6751, 0.6695]}
    for v, q in quoted.items():
        for s, val in zip([42, 1, 2, 3, 4], q):
            check(f'{v} per-seed AUC seed {s}', runs[v][s]['final']['auc'], val, 5e-5)
    # direction of the per-seed comparison quoted in Section 7.1
    below = sum(1 for s in [42, 1, 2, 3, 4]
                if runs['e_best'][s]['final']['auc'] < runs['base'][s]['final']['auc'])
    check('seeds where Full is BELOW the backbone', below, 3)
    check('seeds where Full is ABOVE the backbone', 5 - below, 2)
    if tex:
        in_tex('per-seed direction', tex,
               'on three of the five seeds the full model is marginally below')

    def paired(t, c, m):
        sh = sorted(set(runs[t]) & set(runs[c]))
        x = np.array([runs[t][s]['final'][m] for s in sh])
        y = np.array([runs[c][s]['final'][m] for s in sh])
        return float(x.mean() - y.mean()), float(stats.ttest_rel(x, y).pvalue)

    for t, c, m, dif, pv in (('e_best', 'base', 'auc', 0.0014, 0.66),
                             ('e_best_hyp', 'e_best', 'auc', -0.0060, 0.015),
                             ('e_best_hyp', 'e_best', 'mrr', -0.0058, 0.037),
                             ('e_best_hyp', 'e_best', 'ndcg5', -0.0086, 0.019),
                             ('e_best_hyp', 'e_best', 'ndcg10', -0.0068, 0.020),
                             ('e_best_hyp', 'base', 'auc', -0.0046, 0.27),
                             ('e_collab_large', 'base_large', 'auc', -0.0004, 0.61)):
        d, p = paired(t, c, m)
        check(f'{t} vs {c} {m}: mean diff', d, dif, 5e-5)
        check(f'{t} vs {c} {m}: across-seed paired-t p', p, pv, 0.005)

    def per_imp(v, m):
        acc = defaultdict(list)
        for r in runs[v].values():
            pi = r['per_impression']
            for i, x in zip(pi['impression_ids'], pi[m]):
                acc[i].append(x)
        return np.array([np.nanmean(acc[i]) for i in sorted(acc)])

    for m, pt_w, w_w in (('auc', 0.002, 0.06), ('ndcg10', 0.0009, 0.71)):
        x, y = per_imp('e_best', m), per_imp('base', m)
        ok = ~(np.isnan(x) | np.isnan(y)); d = x[ok] - y[ok]
        check(f'per-impression n ({m})', ok.sum(), 73152)
        check(f'per-impression paired-t p ({m})',
              stats.ttest_rel(x[ok], y[ok]).pvalue, pt_w, 0.0006)
        check(f'per-impression Wilcoxon p ({m})',
              stats.wilcoxon(d[d != 0]).pvalue, w_w, 0.005)

    # the variance claim: descriptive only, robust test must NOT be significant
    b = [runs['base'][s]['final']['auc'] for s in [42, 1, 2, 3, 4]]
    f = [runs['e_best'][s]['final']['auc'] for s in [42, 1, 2, 3, 4]]
    check('std ratio base/Full (threefold claim)',
          np.std(b, ddof=1) / np.std(f, ddof=1), 3.15, 0.15)
    lev = stats.levene(b, f).pvalue
    check('Levene p (must stay non-significant)', lev, 0.21, 0.02)
    (OK if lev > 0.05 else BAD).append(
        ('variance claim kept descriptive (Levene p > 0.05)',
         f'p={lev:.3f}', 'p>0.05', ''))

    # ------------------------------------------------------------- reporting
    print(f'\n{len(OK)} passed, {len(BAD)} failed\n')
    if BAD:
        print('FAILURES')
        for nm, got, wnt, u in BAD:
            print(f'  FAIL  {nm}: got {got}, expected {wnt} {u}')
    else:
        print('All manuscript numbers reproduce from the released artefacts.')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
