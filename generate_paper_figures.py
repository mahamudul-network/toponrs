"""
Publication-quality figures for TopoNRS paper.
Uses actual experimental data. Beta_1 is computed correctly as E - V_active + C_active.
"""
import json, pickle, numpy as np, os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
    'font.size'        : 10,
    'axes.titlesize'   : 10.5,
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 8.5,
    'figure.dpi'       : 300,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.28,
    'grid.linestyle'   : '--',
    'lines.linewidth'  : 1.8,
})

FIG = './figures'
CB  = '#1f77b4'  # blue
CO  = '#ff7f0e'  # orange
CG  = '#2ca02c'  # green
CR  = '#d62728'  # red
CP  = '#9467bd'  # purple
CGR = '#7f7f7f'  # gray

V_TOTAL = 33195   # total news nodes in the graph

# ---- MDPI production style for numbers inside figures -------------------------
# 1. thousands separators on numbers with five or more digits (not four digits);
# 2. scientific notation typeset as 8 x 10^3, never as "8E3"/"1e6".
from matplotlib.ticker import FuncFormatter

def _commas(ax, axis='y'):
    f = FuncFormatter(lambda v, _: f'{int(round(v)):,}' if abs(v) >= 10000
                                   else f'{int(round(v)):d}')
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(f)

def _sci(ax, axis='y'):
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(0, 0), useMathText=True)


def load_betti():
    with open('./topology_data/betti_curves.json') as f:
        raw = json.load(f)
    rows = []
    for d in raw:
        E   = d['num_edges']
        V_a = d['num_vertices']
        C_t = d['beta_0']                    # beta_0 for full graph (correct)
        C_a = C_t - (V_TOTAL - V_a)          # components among active vertices
        b1  = max(0, E - V_a + C_a)          # correct beta_1
        rows.append({
            'threshold' : d['threshold'],
            'beta_0'    : C_t,
            'beta_1'    : b1,
            'num_edges' : E,
            'num_verts' : V_a,
        })
    rows.sort(key=lambda r: r['threshold'])
    return rows


def load_degrees():
    with open('./topology_data/degree_distribution.json') as f:
        raw = json.load(f)
    items = sorted([(int(k), v) for k, v in raw.items()])
    return items


# ── Figure 1: Degree distribution ─────────────────────────────────────────────
def fig_degree_distribution():
    items = load_degrees()
    deg = np.array([d for d, _ in items])
    cnt = np.array([c for _, c in items])

    # Non-zero only for power-law region
    mask  = deg > 0
    dg, ct = deg[mask], cnt[mask]

    # Power-law fit (tail: degree >= 10)
    tm    = dg >= 10
    coef  = np.polyfit(np.log10(dg[tm]), np.log10(ct[tm] + 1e-9), 1)
    gamma = -coef[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Left: scatter + fit
    ax1.scatter(dg, ct, s=5, alpha=0.45, color=CB, linewidths=0, label='Observed')
    ax1.plot(dg[tm], 10 ** np.polyval(coef, np.log10(dg[tm])),
             color=CR, lw=2, ls='--', label=f'Power-law fit  $\\gamma = {gamma:.2f}$')
    ax1.set(xscale='log', yscale='log',
            xlabel='Degree $k$', ylabel='Count  $N(k)$',
            title='(a) Degree distribution')
    ax1.legend(loc='upper right')

    stats = (f'Nodes: {V_TOTAL:,}\nIsolated: 15,637 (47.1%)\n'
             f'Max degree: 146,213\n$\\gamma \\approx {gamma:.2f}$')
    ax1.text(0.03, 0.05, stats, transform=ax1.transAxes, fontsize=8,
             va='bottom', bbox=dict(boxstyle='round,pad=0.35',
             fc='lightyellow', alpha=0.85, ec='#cccc88'))

    # Right: CCDF
    all_d  = np.repeat(dg, ct)
    srt    = np.sort(all_d)
    ccdf   = 1.0 - np.arange(len(srt)) / len(srt)
    ax2.plot(srt, ccdf, color=CB, lw=1.6)

    tm2 = srt >= 10
    c2  = np.polyfit(np.log10(srt[tm2]), np.log10(ccdf[tm2] + 1e-12), 1)
    ax2.plot(srt[tm2], 10 ** np.polyval(c2, np.log10(srt[tm2])),
             color=CR, lw=2, ls='--', label=f'Tail fit  $\\beta = {-c2[0]:.2f}$')
    ax2.set(xscale='log', yscale='log',
            xlabel='Degree $k$',
            ylabel='$P(K \\geq k)$',
            title='(b) Complementary CDF')
    ax2.legend()

    plt.tight_layout(pad=1.2)
    _save('degree_distribution')


# ── Figure 2: Betti curves (corrected) ────────────────────────────────────────
def fig_betti_curves():
    rows = load_betti()
    thr  = np.array([r['threshold']  for r in rows])
    b0   = np.array([r['beta_0']     for r in rows])
    b1   = np.array([r['beta_1']     for r in rows])
    E    = np.array([r['num_edges']  for r in rows])
    Va   = np.array([r['num_verts']  for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))

    # (a) beta_0
    ax = axes[0]
    ax.plot(thr, b0, 'o-', color=CB, ms=7, label='$\\beta_0$', zorder=3)
    ax.fill_between(thr, b0, V_TOTAL, alpha=0.08, color=CB)
    ax.axhline(V_TOTAL, color=CGR, lw=0.9, ls=':', alpha=0.6)
    ax.set(xlabel='Co-click threshold $\\varepsilon$',
           ylabel='$\\beta_0$ — connected components',
           title='(a) $\\beta_0$ across filtration')
    ax.invert_xaxis()
    ax.annotate(f'{b0[0]:,}', xy=(thr[0], b0[0]), xytext=(-10, 14),
                textcoords='offset points', fontsize=7.5, color=CB,
                ha='right', va='bottom')
    ax.annotate(f'{b0[-1]:,}', xy=(thr[-1], b0[-1]), xytext=(14, -16),
                textcoords='offset points', fontsize=7.5, color=CB,
                ha='left', va='top')
    ax.margins(x=0.07, y=0.13)
    _commas(ax)

    # (b) beta_1 (corrected)
    ax = axes[1]
    ax.plot(thr, b1, 's-', color=CO, ms=7, label='$\\beta_1$ (corrected)', zorder=3)
    ax.fill_between(thr, b1, alpha=0.10, color=CO)
    ax.set(xlabel='Co-click threshold $\\varepsilon$',
           ylabel='$\\beta_1$ — independent 1-cycles',
           title='(b) $\\beta_1$ across filtration')
    ax.invert_xaxis()
    # Annotate the rapid cycle formation
    ax.annotate('Cycles form\nrapidly here',
                xy=(thr[-2], b1[-2]), xycoords='data',
                xytext=(0.34, 0.42), textcoords='axes fraction',
                fontsize=8, color=CO, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color=CO, lw=1.2,
                                shrinkA=2, shrinkB=4))
    ax.margins(y=0.12)
    _sci(ax)

    # (c) Euler characteristic
    ax = axes[2]
    chi = b0 - b1
    ax.plot(thr, chi, '^-', color=CG, ms=7, label='$\\chi = \\beta_0 - \\beta_1$', zorder=3)
    ax.fill_between(thr, chi, alpha=0.10, color=CG)
    ax.axhline(0, color=CGR, lw=0.9, ls=':')
    ax.set(xlabel='Co-click threshold $\\varepsilon$',
           ylabel='Euler characteristic $\\chi$',
           title='(c) Euler characteristic')
    ax.invert_xaxis()
    _sci(ax)

    # Second y-axis: edge count
    ax3r = axes[2].twinx()
    ax3r.plot(thr, E, 'x--', color=CP, ms=6, lw=1.2, alpha=0.7, label='Edge count')
    ax3r.set_ylabel('Active edges', color=CP)
    ax3r.tick_params(axis='y', labelcolor=CP)
    ax3r.spines['right'].set_visible(True)
    _sci(ax3r)
    ax3r.yaxis.get_offset_text().set_color(CP)

    handles = [plt.Line2D([0],[0], color=CG, marker='^', ms=6, label='$\\chi$'),
               plt.Line2D([0],[0], color=CP, marker='x', ms=6, ls='--', label='Edge count')]
    axes[2].legend(handles=handles, fontsize=8, loc='center left',
                   framealpha=0.92, borderpad=0.4)

    plt.tight_layout(pad=1.2)
    _save('betti_curves')


# ── Figure 3: Local graph topology features ────────────────────────────────────
def fig_graph_features():
    with open('./topology_data/news_ph_features.pkl', 'rb') as f:
        ph = pickle.load(f)

    feat_labels = [
        ('Degree $k_i$',                    'log$(1+k_i)$'),
        ('Mean co-click weight $\\bar{w}_i$','log$(1+\\bar{w}_i)$'),
        ('Weight std $\\sigma_i$',           'log$(1+\\sigma_i)$'),
        ('Max weight $w_i^{\\max}$',         'log$(1+w_i^{\\max})$'),
        ('Min weight $w_i^{\\min}$',         'log$(1+w_i^{\\min})$'),
        ('Clustering coeff. $c_i$',          '$c_i$'),
    ]

    feats = np.array(list(ph.values()), dtype=float)
    nz    = feats.sum(axis=1) > 0
    F     = feats[nz]
    pct   = 100.0 * nz.sum() / len(feats)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    colors = [CB, CO, CG, CR, CP, CGR]

    for i, ax in enumerate(axes.flat):
        vals = F[:, i]
        pos  = vals[vals > 0]
        if len(pos) == 0:
            ax.text(0.5, 0.5, 'All zero', ha='center', va='center',
                    transform=ax.transAxes, color=CGR)
            ax.set_title(f'({chr(97+i)}) {feat_labels[i][0]}', fontsize=9.5)
            continue
        v = np.log1p(pos) if i < 5 else pos
        ax.hist(v, bins=45, color=colors[i], alpha=0.78,
                edgecolor='white', linewidth=0.4)
        ax.set_xlabel(feat_labels[i][1], fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        _commas(ax)
        ax.set_title(f'({chr(97+i)}) {feat_labels[i][0]}', fontsize=9.5)
        ax.text(0.97, 0.93,
                f'non-zero: {100*len(pos)/len(feats):.1f}%\nmedian: {np.median(pos):.1f}',
                transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

    fig.suptitle(
        f'Local co-click graph topology features  '
        f'({pct:.1f}% of {V_TOTAL:,} news articles have at least one edge)',
        fontsize=10.5, y=1.005)
    plt.tight_layout(pad=1.0)
    _save('graph_features')


# ── Figure 4: Training convergence ────────────────────────────────────────────
def fig_convergence():
    with open('./results/TopoNRS_results.json') as f:
        raw = json.load(f)

    rows = [r for r in raw if isinstance(r.get('step'), int)]
    steps  = [r['step']   for r in rows]
    auc    = [r['auc']    for r in rows]
    mrr    = [r['mrr']    for r in rows]
    nd5    = [r['ndcg5']  for r in rows]
    nd10   = [r['ndcg10'] for r in rows]
    loss   = [r['loss']   for r in rows]

    nrms = {'AUC': 0.6696, 'MRR': 0.3155, 'nDCG@5': 0.3493, 'nDCG@10': 0.4112}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    panels = [
        (auc,  'AUC',     CB, nrms['AUC'],     True),
        (mrr,  'MRR',     CO, nrms['MRR'],     True),
        (nd5,  'nDCG@5',  CG, nrms['nDCG@5'],  True),
        (nd10, 'nDCG@10', CR, nrms['nDCG@10'], True),
        (loss, 'Training loss', CP, None, False),
    ]

    def moving_avg(x, w=5):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    for i, (vals, label, color, ref, higher_better) in enumerate(panels):
        ax = axes.flat[i]
        ax.plot(steps, vals, color=color, lw=1.3, alpha=0.55, label='Per-step')
        sm = moving_avg(vals)
        ax.plot(steps[2:2+len(sm)], sm, color=color, lw=2.2, label='5-step avg')
        if ref is not None:
            ax.axhline(ref, color='black', ls='--', lw=1.2, alpha=0.65,
                       label=f'NRMS: {ref:.4f}')
        best_i = np.argmax(vals) if higher_better else np.argmin(vals)
        ax.scatter([steps[best_i]], [vals[best_i]], s=80, color=color,
                   zorder=5, marker='*')
        ax.annotate(f'  {vals[best_i]:.4f}',
                    xy=(steps[best_i], vals[best_i]),
                    fontsize=8, color=color, va='center')
        ax.set(xlabel='Training step', ylabel=label,
               title=f'({chr(97+i)}) {label}')
        ax.legend(fontsize=7.5,
                  loc='lower right' if higher_better else 'upper right')
        # Epoch boundaries (approx)
        for ep_s, ep_l in [(3200, 'ep.1→2'), (6400, 'ep.2→3')]:
            if ep_s <= max(steps):
                ax.axvline(ep_s, color=CGR, lw=0.7, ls=':', alpha=0.5)
                ax.text(ep_s, ax.get_ylim()[0], ep_l, fontsize=6.5,
                        color=CGR, ha='center', va='bottom')

    # Last panel: note on full training
    ax_note = axes.flat[5]
    ax_note.axis('off')
    txt = (
        "These results correspond to approximately\n"
        "two full training epochs on MIND-small.\n\n"
        "The model had not yet converged at this\n"
        "checkpoint. Full results (6 epochs) are\n"
        "pending and will replace the placeholder\n"
        "entries in the main comparison table.\n\n"
        "NRMS baseline (confirmed, this run):\n"
        "  AUC  = 0.6696\n"
        "  MRR  = 0.3155\n"
        "  nDCG@5  = 0.3493\n"
        "  nDCG@10 = 0.4112"
    )
    ax_note.text(0.05, 0.95, txt, transform=ax_note.transAxes,
                 fontsize=9, va='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', fc='#f8f8f8',
                           ec='#cccccc', alpha=0.9))
    ax_note.set_title('(f) Training status note', fontsize=10.5)

    plt.tight_layout(pad=1.3)
    _save('training_convergence')


# ── Figure 5: Poincare ball intuition ─────────────────────────────────────────
def fig_poincare():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))

    # ── Left: Poincare disk ───────────────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 400)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', lw=1.4, alpha=0.5)
    for r in [0.33, 0.6, 0.82]:
        ax1.add_patch(plt.Circle((0,0), r, fill=False, color=CGR,
                                  lw=0.6, ls=':', alpha=0.45))
    ax1.set_xlim(-1.18, 1.18); ax1.set_ylim(-1.18, 1.18)
    ax1.set_aspect('equal'); ax1.axis('off')

    np.random.seed(7)
    # Popular articles: near center
    pr = np.random.uniform(0, 0.28, 5); pt = np.random.uniform(0,2*np.pi,5)
    ax1.scatter(pr*np.cos(pt), pr*np.sin(pt), s=130, color=CO, zorder=5,
                marker='*', edgecolors='k', lw=0.5, label='Broadly popular')
    # Niche articles: near boundary
    nr = np.random.uniform(0.72, 0.90, 16); nt = np.random.uniform(0,2*np.pi,16)
    ax1.scatter(nr*np.cos(nt), nr*np.sin(nt), s=45, color=CB, zorder=5,
                marker='o', edgecolors='k', lw=0.4, alpha=0.82, label='Niche articles')
    # User
    ux, uy = 0.52*np.cos(1.05), 0.52*np.sin(1.05)
    ax1.scatter([ux], [uy], s=200, color=CR, zorder=6, marker='D',
                edgecolors='k', lw=0.8, label='User')
    ax1.annotate('', xy=(nr[4]*np.cos(nt[4]), nr[4]*np.sin(nt[4])),
                 xytext=(ux, uy),
                 arrowprops=dict(arrowstyle='->', color=CR, lw=1.5,
                                 connectionstyle='arc3,rad=0.25'))
    ax1.text(0.72, 0.90, 'short hyperbolic\ndistance', ha='center', va='center',
             fontsize=7, color=CR)

    ax1.text(-0.62, -1.12, 'Boundary  $\\|\\mathbf{x}\\| \\to 1$',
              ha='center', fontsize=8, color=CGR)
    # Label the centre from outside the marker cloud, with a leader line, so it
    # no longer sits on top of the star markers.
    ax1.annotate('Centre: broadly\nread content',
                 xy=(0.0, 0.0), xytext=(-0.60, 0.42),
                 ha='center', va='center', fontsize=7.5, color=CO,
                 arrowprops=dict(arrowstyle='-', color=CO, lw=0.8, alpha=0.8,
                                 shrinkA=2, shrinkB=14))
    ax1.legend(loc='lower right', fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(1.16, -0.02), borderpad=0.4)
    ax1.set_title('(a) Poincaré ball embedding intuition', fontsize=10.5, pad=6)

    # ── Right: filtration cartoon ─────────────────────────────────────────────
    ax2.axis('off')
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 5)
    ax2.set_title('(b) Co-click graph at increasing density', fontsize=10.5, pad=6)

    stages = [
        (1.2, 'High $\\varepsilon$\n(sparse)',  [(1.0,3.5),(1.5,2.5)],
         [(0,1)],                                ''),
        (4.0, 'Medium $\\varepsilon$',           [(3.5,3.5),(4.0,2.5),(4.5,3.5)],
         [(0,1),(1,2),(0,2)],                    '$\\beta_1=1$'),
        (7.2, 'Low $\\varepsilon$\n(dense)',     [(6.7,3.6),(7.2,2.5),(7.7,3.6),(7.2,4.2)],
         [(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)],  '$\\beta_1=3$'),
    ]
    for cx, label, nodes, edges, annot in stages:
        for (i,j) in edges:
            ax2.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
                     '-', color=CGR, lw=1.4, alpha=0.6, zorder=1)
        for (nx, ny) in nodes:
            ax2.scatter([nx], [ny], s=80, color=CB, zorder=3,
                        edgecolors='k', lw=0.5)
        ax2.text(cx, 1.85, label, ha='center', fontsize=8.5)
        if annot:
            ax2.text(cx, 4.55, annot, ha='center', fontsize=9,
                     color=CO, fontweight='bold')

    ax2.annotate('', xy=(5.9,3.1), xytext=(5.1,3.1),
                 arrowprops=dict(arrowstyle='->', color='k', lw=1.3))
    ax2.annotate('', xy=(2.8,3.1), xytext=(2.1,3.1),
                 arrowprops=dict(arrowstyle='->', color='k', lw=1.3))
    ax2.text(4.65, 0.95, 'decreasing co-click threshold $\\varepsilon$',
             ha='center', fontsize=8, color='k')

    plt.tight_layout(pad=1.2)
    _save('poincare_intuition')


def _save(name):
    for d in (FIG, './submission/figures'):
        os.makedirs(d, exist_ok=True)
        for ext in ('pdf', 'png'):
            plt.savefig(f'{d}/{name}.{ext}', format=ext)
    plt.close()
    kb = os.path.getsize(f'{FIG}/{name}.pdf') // 1024
    print(f'  {name}.pdf  ({kb} KB)')


if __name__ == '__main__':
    print('Generating figures...')
    fig_degree_distribution()
    fig_betti_curves()
    fig_graph_features()
    fig_convergence()
    fig_poincare()
    print('Done.')
