"""Regenerate figures/training_convergence.pdf from the real logged
validation trajectories (seed 42): NRMS backbone vs TopoNRS."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def traj(v):
    d = json.load(open(f'results/seed_runs/{v}_seed42_results.json'))
    log = [e for e in d['log'] if isinstance(e.get('step'), int)]
    return log

base = traj('base'); topo = traj('e_best')
steps_b = [e['step'] for e in base]; steps_t = [e['step'] for e in topo]

panels = [('auc', 'AUC'), ('mrr', 'MRR'), ('ndcg5', 'nDCG@5'),
          ('ndcg10', 'nDCG@10'), ('loss', 'Training loss')]
fig, axes = plt.subplots(1, 5, figsize=(19, 3.4))
for ax, (key, label) in zip(axes, panels):
    ax.plot(steps_b, [e[key] for e in base], '--', color='#888', label='NRMS')
    ax.plot(steps_t, [e[key] for e in topo], '-', color='#1f77b4', label='TopoNRS')
    ax.set_xlabel('training step'); ax.set_title(label)
    ax.grid(alpha=0.3)
axes[0].legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('submission/figures/training_convergence.pdf', bbox_inches='tight')
print('wrote submission/figures/training_convergence.pdf')
