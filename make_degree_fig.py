"""Regenerate figures/degree_distribution.pdf using the TRUE node degree
(number of co-click neighbours), not the weighted strength. Two panels:
(a) log-log degree histogram with a descriptive heavy-tail fit;
(b) complementary CDF."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('topology_data/degree_true_mindsmall.json'))
deg = np.array(d['degree']); cnt = np.array(d['count'])
p = cnt / cnt.sum()
maxdeg, meandeg, iso = d['max_degree'], d['mean_degree'], d['isolated']

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

# (a) log-log with descriptive slope over deg>=10
ax[0].loglog(deg, p, 'o', ms=3, color='#3b6ea5', alpha=0.7)
m = deg >= 10
a, b = np.polyfit(np.log(deg[m]), np.log(p[m]), 1)
r2 = 1 - ((np.log(p[m]) - (a*np.log(deg[m])+b))**2).sum() / ((np.log(p[m]) - np.log(p[m]).mean())**2).sum()
xs = np.array([10, deg.max()])
ax[0].loglog(xs, np.exp(b) * xs**a, '--', color='#c0392b', lw=2,
             label=f'heavy-tail slope $\\approx{-a:.2f}$ ($R^2={r2:.2f}$)')
ax[0].set_xlabel('degree $k$ (number of co-click neighbours)')
ax[0].set_ylabel('$P(k)$')
ax[0].set_title('(a) True degree distribution (log--log)', fontweight='bold')
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, which='both')
ax[0].text(0.96, 0.95, f'max degree $={maxdeg:,}$\nmean $={meandeg:.0f}$\n'
           f'isolated $=47.1\\%$', transform=ax[0].transAxes, ha='right', va='top',
           fontsize=9, bbox=dict(boxstyle='round', fc='#f0f0f0'))

# (b) CCDF
order = np.argsort(deg)
dd = deg[order]; pp = p[order]
ccdf = 1 - np.cumsum(pp) + pp
ax[1].loglog(dd, ccdf, '-', color='#3b6ea5', lw=1.8)
ax[1].set_xlabel('degree $k$'); ax[1].set_ylabel('$P(K \\geq k)$')
ax[1].set_title('(b) Complementary CDF', fontweight='bold')
ax[1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('submission/figures/degree_distribution.pdf', bbox_inches='tight')
print(f'wrote degree figure: max degree {maxdeg}, slope {-a:.3f}, R2 {r2:.3f}')
