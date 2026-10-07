"""MLP corruption retention: one row of three panels, one per dataset.

For OOD-first runs, accuracy on the phase-1 corruption set, measured twice:
  pale bar  = end of phase 1 (the corruption mapping is memorized)
  dark bar  = end of phase 2 (after the target phase)
x is model scale (parameter count); bars smaller than 6% of the axis carry their value as text.
Reads results/curriculum.jsonl. Output: results/figures/mlp_corruption_retention.png
"""
import os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, FIGDIR

CONFIG = 'mlp'
PANELS = [('mlp', 'mnist', 'MLP — MNIST'), ('mlp', 'fashion', 'MLP — Fashion-MNIST'), ('mlp', 'kmnist', 'MLP — KMNIST')]
COLORS = ['#4E9BD5', '#E6B422', '#3FB68B']


def pfmt(p):
    return '%.0fM' % (p / 1e6) if p >= 1e6 else ('%.0fK' % (p / 1e3) if p >= 1e3 else '%d' % p)


recs = [r for r in load('curriculum.jsonl') if r.get('arch') == 'mlp']
mem = defaultdict(list); sur = defaultdict(list); par = {}
for r in recs:
    if r.get('config') != CONFIG or r['ord'] != 'junkfirst':
        continue
    key = (r['arch'], r['task'], r['width'])
    if r.get('junk_bias') is not None: mem[key].append(r['junk_bias'])
    if r.get('junk_end') is not None: sur[key].append(r['junk_end'])
    par[key] = r['params_M'] * 1e6

fig, AX = plt.subplots(1, 3, figsize=(17, 4.5), squeeze=False)
for pi, (arch, task, title) in enumerate(PANELS):
    ax = AX[pi // 3][pi % 3]
    col = COLORS[pi % 3]
    widths = sorted({k[2] for k in mem if k[:2] == (arch, task)}, key=lambda W: par[(arch, task, W)])
    if not widths:
        ax.set_axis_off(); continue
    xs = np.arange(len(widths))
    p1 = [float(np.mean(mem[(arch, task, W)])) for W in widths]
    p2 = [float(np.mean(sur[(arch, task, W)])) if sur.get((arch, task, W)) else np.nan for W in widths]
    ax.bar(xs, p1, width=0.78, color=col, alpha=0.38, edgecolor=col, lw=0.8)      # end of phase 1 (pale)
    ax.bar(xs, p2, width=0.42, color=col, edgecolor='none')                        # end of phase 2 (dark)
    for x, v in zip(xs, p2):
        if np.isfinite(v) and v < 0.06:                                            # label the tiny bars
            ax.text(x, 0.012, '%.3f' % v, rotation=90, ha='center', va='bottom', fontsize=7, color=col)
    ax.set_xticks(xs); ax.set_xticklabels([pfmt(par[(arch, task, W)]) for W in widths], rotation=45, fontsize=8.5)
    ax.set_ylim(0, 1.06); ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', color='#eee', lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    if pi % 3 == 0: ax.set_ylabel('corruption-set accuracy', fontsize=10.5)
    ax.set_xlabel('model scale (parameters)', fontsize=10.5)

fig.legend([Patch(facecolor='#9e9e9e', alpha=0.38, edgecolor='#9e9e9e'), Patch(facecolor='#555')],
           ['end of phase 1  (corruption memorized)', 'end of phase 2  (after target phase)'],
           loc='lower center', ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.12, 1, 1], h_pad=2.0)
os.makedirs(FIGDIR, exist_ok=True)
out = os.path.join(FIGDIR, 'mlp_corruption_retention.png'); fig.savefig(out, dpi=150); print('SAVED', out)
