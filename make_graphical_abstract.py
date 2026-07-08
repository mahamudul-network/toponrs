"""Regenerate the graphical abstract with an HONEST panel (d):
parity in mean + ~3x variance reduction over NRMS (5 seeds), replacing
the retracted single-seed 'best on every metric' claim.
Panels (a) degree distribution, (b) Betti curves, (c) Poincare schematic
are reproduced from saved topology data."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

N = 33195
dd = {int(k): v for k, v in json.load(open('topology_data/degree_distribution.json')).items()}
bc = json.load(open('topology_data/betti_curves.json'))

fig = plt.figure(figsize=(11, 11))
fig.suptitle("Co-Click Graph Topology of MIND-small\nand a Topology-Informed News Recommender",
             fontsize=17, fontweight='bold', y=0.98)

# (a) degree distribution
ax = fig.add_subplot(2, 2, 1)
ks = np.array(sorted(k for k in dd if k > 0)); ps = np.array([dd[k] for k in ks], float)
ps = ps / ps.sum()
ax.loglog(ks, ps, 'o', ms=3, color='#3b6ea5')
m = ks >= 10
c = np.polyfit(np.log(ks[m]), np.log(ps[m]), 1)
ax.loglog(ks[m], np.exp(np.polyval(c, np.log(ks[m]))), '--', color='#c0392b',
          label=r'power-law fit $\gamma\approx1.1$')
ax.set_xlabel('degree $k$'); ax.set_ylabel('$P(k)$')
ax.set_title('(a) Heavy-tailed co-click degrees', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
ax.text(0.95, 0.95, '$k_{max}$=146,213\n33,195 articles\n$3.18\\times10^6$ edges',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='#f0f0f0'))

# (b) Betti curves
ax = fig.add_subplot(2, 2, 2)
th = [e['threshold'] for e in bc]
b0 = [e['beta_0'] for e in bc]
b1 = [max(0, e['num_edges'] + e['beta_0'] - N) for e in bc]
x = range(len(th))
ax.plot(x, b0, 'o-', color='#3b6ea5', label=r'$\beta_0$')
ax.set_ylabel(r'$\beta_0$ (components)', color='#3b6ea5')
ax2 = ax.twinx()
ax2.plot(x, np.array(b1) + 1, 's-', color='#e07b1a', label=r'$\beta_1$')
ax2.set_yscale('symlog'); ax2.set_ylabel(r'$\beta_1$ (cycles, symlog)', color='#e07b1a')
ax.set_xticks(list(x)); ax.set_xticklabels([int(t) for t in th], fontsize=7)
ax.set_xlabel(r'co-click threshold $\varepsilon$ (high $\to$ low)')
ax.set_title(r'(b) Betti curves: forest $\to$ cycle-rich', fontweight='bold')
ax.grid(alpha=0.3)

# (c) Poincare schematic
ax = fig.add_subplot(2, 2, 3)
circ = plt.Circle((0, 0), 1, fill=False, color='k', lw=1.5); ax.add_patch(circ)
rng = np.random.default_rng(0)
ang = rng.uniform(0, 2*np.pi, 40); rad = rng.uniform(0.75, 0.98, 40)
ax.scatter(rad*np.cos(ang), rad*np.sin(ang), s=18, color='#3b6ea5', label='niche articles')
ha = rng.uniform(0, 2*np.pi, 6); hr = rng.uniform(0, 0.35, 6)
ax.scatter(hr*np.cos(ha), hr*np.sin(ha), marker='*', s=180, color='#e8a33d',
           edgecolor='k', label='popular hubs')
ax.scatter([0.72], [0.4], marker='D', s=90, color='#b03030', label='user')
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.25, 1.1); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('(c) Hyperbolic embedding (optional)', fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.text(0, -1.18, u'Poincaré ball $\\mathbb{B}^d_c$', ha='center', fontsize=10)

# (d) HONEST result -- horizontal bars with labels/values outside the bars
ax = fig.add_subplot(2, 2, 4); ax.axis('off')
ax.set_title('(d) Honest result (5 seeds)', fontweight='bold', loc='center')
axb = ax.inset_axes([0.18, 0.55, 0.78, 0.34])
labels = ['NRMS', 'TopoNRS']
means = [0.6745, 0.6759]; stds = [0.0074, 0.0023]
ypos = [1, 0]
axb.barh(ypos, means, xerr=stds, height=0.55,
         color=['#8aa9c9', '#b03030'], error_kw=dict(ecolor='k', capsize=5))
for yi, mu, sd in zip(ypos, means, stds):
    axb.text(mu + sd + 0.0008, yi, f'{mu:.4f}$\\pm${sd:.4f}', va='center',
             ha='left', fontsize=9)
axb.set_yticks(ypos); axb.set_yticklabels(labels, fontsize=10)
axb.set_xlim(0.665, 0.692); axb.set_xlabel('AUC (val., 5 seeds)', fontsize=9)
axb.spines[['top', 'right']].set_visible(False)
ax.text(0.5, 0.24,
        'AUC parity in mean (paired $p=0.66$)\n'
        r'but $\approx$3$\times$ lower seed variance (std 0.0074$\to$0.0023).'
        '\nCo-click topology acts as a stabilising prior,\nnot an accuracy gain.',
        transform=ax.transAxes, ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', fc='#fdf3e7', ec='#e07b1a'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('submission/graphical_abstract.png', dpi=200, bbox_inches='tight')
plt.savefig('submission/graphical_abstract.pdf', bbox_inches='tight')
print('wrote submission/graphical_abstract.{png,pdf}')
