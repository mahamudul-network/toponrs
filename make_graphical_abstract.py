"""Graphical abstract: topological study of news co-click graphs + honest
negative result. Panels: (a) TRUE degree distribution, (b) Betti curves,
(c) flag-complex PH finding, (d) parity across two datasets."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 33195
dd = json.load(open('topology_data/degree_true_mindsmall.json'))
deg = np.array(dd['degree']); cnt = np.array(dd['count'])
bc = json.load(open('topology_data/betti_curves.json'))

fig = plt.figure(figsize=(11, 11))
fig.suptitle("Persistent Homology of News Co-Click Graphs\n"
             "and a Negative Result for Topology-Informed Recommendation",
             fontsize=15, fontweight='bold', y=0.99)

# (a) TRUE degree distribution
ax = fig.add_subplot(2, 2, 1)
p = cnt / cnt.sum()
ax.loglog(deg, p, 'o', ms=3, color='#3b6ea5', alpha=0.7)
m = deg >= 10
a, b = np.polyfit(np.log(deg[m]), np.log(p[m]), 1)
ax.loglog(deg[m], np.exp(b)*deg[m]**a, '--', color='#c0392b',
          label=r'heavy-tail slope $\approx0.85$')
ax.set_xlabel('degree $k$ (co-click neighbours)'); ax.set_ylabel('$P(k)$')
ax.set_title('(a) True degree distribution', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
ax.text(0.95, 0.95, 'max degree = 9,321\nmean = 191\n33,195 articles',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='#f0f0f0'))

# (b) Betti curves (1-skeleton)
ax = fig.add_subplot(2, 2, 2)
th = [e['threshold'] for e in bc]
b0 = [e['beta_0'] for e in bc]
b1 = [max(0, e['num_edges'] + e['beta_0'] - N) for e in bc]
x = range(len(th))
ax.plot(x, b0, 'o-', color='#3b6ea5', label=r'$\beta_0$')
ax.set_ylabel(r'$\beta_0$ (components)', color='#3b6ea5')
ax2 = ax.twinx()
ax2.plot(x, np.array(b1)+1, 's-', color='#e07b1a', label=r'$\beta_1$')
ax2.set_yscale('symlog'); ax2.set_ylabel(r'$\beta_1$ (1-skeleton, symlog)', color='#e07b1a')
ax.set_xticks(list(x)); ax.set_xticklabels([int(t) for t in th], fontsize=7)
ax.set_xlabel(r'co-click threshold $\varepsilon$ (high $\to$ low)')
ax.set_title('(b) Betti curves (graph 1-skeleton)', fontweight='bold')
ax.grid(alpha=0.3)

# (c) flag-complex PH finding
ax = fig.add_subplot(2, 2, 3); ax.axis('off')
ax.set_title('(c) Genuine persistent homology (GUDHI)', fontweight='bold')
ax.text(0.5, 0.62,
        r'Flag (clique) complex, expanded to tetrahedra:',
        transform=ax.transAxes, ha='center', fontsize=11)
ax.text(0.5, 0.40,
        r'$\beta_1 = \beta_2 = 0$  on both datasets',
        transform=ax.transAxes, ha='center', fontsize=15, color='#b03030',
        fontweight='bold')
ax.text(0.5, 0.16,
        '1-skeleton cycles are filled by triangles\n'
        '(clique-like cores); the genuine\n'
        'topological signal is in $H_0$ (communities).',
        transform=ax.transAxes, ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', fc='#eef3f8', ec='#3b6ea5'))

# (d) parity across two datasets
ax = fig.add_subplot(2, 2, 4); ax.axis('off')
ax.set_title('(d) Negative result, two datasets', fontweight='bold')
axb = ax.inset_axes([0.16, 0.52, 0.80, 0.36])
groups = ['MIND-small', 'MIND-large']
nrms = [0.6745, 0.6907]; topo = [0.6759, 0.6903]
y = np.arange(2); h = 0.35
axb.barh(y+h/2, nrms, h, color='#8aa9c9', label='NRMS')
axb.barh(y-h/2, topo, h, color='#b03030', label='TopoNRS')
axb.set_yticks(y); axb.set_yticklabels(groups, fontsize=9)
axb.set_xlim(0.66, 0.70); axb.set_xlabel('AUC', fontsize=9)
axb.legend(fontsize=8, loc='lower right'); axb.spines[['top','right']].set_visible(False)
ax.text(0.5, 0.22,
        'TopoNRS matches NRMS in mean accuracy\n'
        r'($p=0.66$ small, $p=0.61$ large): a strong content'
        '\nencoder already captures the signal.',
        transform=ax.transAxes, ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', fc='#fdf3e7', ec='#e07b1a'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('submission/graphical_abstract.png', dpi=200, bbox_inches='tight')
plt.savefig('submission/graphical_abstract.pdf', bbox_inches='tight')
print('wrote submission/graphical_abstract.{png,pdf}')
