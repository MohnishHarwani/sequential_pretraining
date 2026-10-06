"""Foundation figure: 2 rows x 2 columns of grouped bars, ONE FIGURE PER MODEL SCALE.

  columns: the two target corpora (FineWeb, TinyStories)
  rows:    English (target) validation loss, and aggregate loss over both distributions
  groups:  the four OOD corpora; within each group three arms --
           English-first (grey), OOD-first (solid colour), OOD-first + Exposure Therapy (dotted)

Runs are separated by scale (the `config` field: 100M / 500M / 1B); a bar never mixes scales.
Bars are the mean over seeds, black dots are individual runs, and each bar carries its value to
4 decimals. Reads results/foundation.jsonl.
Output: results/figures/foundation_<scale>.png, one per scale present in the results.

    python figures/fig_foundation.py                 # every scale found
    python figures/fig_foundation.py --scale 1B      # just one
"""
import argparse
import os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, FIGDIR

GOODS = [('fineweb', 'FineWeb'), ('tinystories', 'TinyStories')]
OODS = [('code', 'code', 'CodeSearchNet', '#c0392b'), ('math', 'math', 'OpenWebMath', '#e08214'),
        ('dewiki', 'German', 'German Wikipedia', '#1b7a4b'), ('fiwiki', 'Finnish', 'Finnish Wikipedia', '#6a3d9a')]
ARMS = [(('goodfirst', 0.0), None), (('junkfirst', 0.0), None), (('junkfirst', 0.30), '...')]

ap = argparse.ArgumentParser()
ap.add_argument('--scale', default=None, help='plot only this scale (default: one figure per scale)')
ap.add_argument('--input', default='foundation.jsonl')
ap.add_argument('--out-dir', default=FIGDIR)
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--preliminary', action='store_true', help='Label incomplete results and show seed counts')
ap.add_argument('--snapshot-label', default='')
ap.add_argument('--expected-runs', type=int, default=72)
args = ap.parse_args()
FIGDIR = args.out_dir

recs = load(args.input)
D = defaultdict(list)                       # (scale, good, ood, arm) -> [(val_ce, aggregate) per seed]
for r in recs:
    f = r.get('final', {})
    if f.get('val_ce') is None:
        continue
    agg = f.get('aggregate', (f['val_ce'] + f.get('ood_val_ce', f['val_ce'])) / 2.0)
    good = r['good'].split('_')[0]; ood = r['jcorpus'].split('_')[0]
    D[(r['config'], good, ood, (r['ord'], round(r['exposure_rate'], 3)))].append((f['val_ce'], agg))

SCALES = sorted({k[0] for k in D}, key=lambda s: {'100M': 0, '500M': 1, '1B': 2}.get(s, 99))
if args.scale:
    SCALES = [s for s in SCALES if s == args.scale] or [args.scale]


def values(scale, good, ood, arm, mode):
    return [(v if mode == 'mem' else a) for v, a in D.get((scale, good, ood, arm), [])]


def draw(ax, scale, good, mode, show_groups):
    xs, ys, cs, hs, dots, groups = [], [], [], [], [], []
    missing = []
    x = 0
    for ocode, olab, osrc, ocol in OODS:
        x0 = x
        for arm, hatch in ARMS:
            vv = values(scale, good, ocode, arm, mode)
            if not vv:
                missing.append(x)
                x += 1; continue
            xs.append(x); ys.append(float(np.mean(vv)))
            cs.append('#9e9e9e' if arm[0] == 'goodfirst' else ocol); hs.append(hatch)
            dots.append((x, vv)); x += 1
        groups.append(((x0 + x - 1) / 2.0, f'{olab}\n({osrc})')); x += 0.6
    if not dots:
        ax.set_axis_off(); return
    for xi, yi, ci, hi in zip(xs, ys, cs, hs):
        ax.bar(xi, yi, color=ci, edgecolor='k', lw=.6, hatch=hi, alpha=.55 if hi else 1.0)
    for xi, vv in dots:
        ax.scatter([xi] * len(vv), vv, s=13, c='k', zorder=5)
    for xi, yi in zip(xs, ys):
        ax.annotate('%.4f' % yi, (xi, yi), fontsize=6.5, ha='center', xytext=(0, 3), textcoords='offset points')
    lo = min(min(vv) for _, vv in dots); hi_ = max(max(vv) for _, vv in dots)
    pad = (hi_ - lo) * 0.18 or 0.01
    ax.set_ylim(lo - pad, hi_ + pad)
    if args.preliminary:
        for xi, vv in dots:
            ax.text(xi, .015, f'n={len(vv)}', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
        for xi in missing:
            ax.text(xi, .03, 'pending', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', rotation=90, fontsize=8, color='#666666')
    ax.set_xticks([g[0] for g in groups])
    ax.set_xticklabels([g[1] for g in groups] if show_groups else [], fontsize=10.5)
    ax.tick_params(axis='x', length=0); ax.grid(axis='y', alpha=.3)
    ax.set_ylabel('English Loss (nats/byte)' if mode == 'mem' else 'Aggregate Loss (nats/byte)')


os.makedirs(FIGDIR, exist_ok=True)
for scale in SCALES:
    fig, AX = plt.subplots(2, 2, figsize=(17, 13.0))
    for ri, mode in enumerate(('mem', 'joint')):
        for ci, (good, glab) in enumerate(GOODS):
            draw(AX[ri][ci], scale, good, mode, show_groups=(ri == 1))
            if ri == 0:
                AX[ri][ci].set_title(glab, fontsize=11, fontweight='bold')
    cols = [c for _, _, _, c in OODS]
    h_eng = Patch(facecolor='#9e9e9e', edgecolor='k', lw=.6)
    h_ood = tuple(Patch(facecolor=c, edgecolor='k', lw=.6) for c in cols)
    h_exp = tuple(Patch(facecolor=c, edgecolor='k', lw=.6, hatch='...', alpha=.55) for c in cols)
    fig.legend([h_eng, h_ood, h_exp], ['English-first', 'OOD-first', 'OOD-first + Exposure Therapy'],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, loc='lower center', ncol=3,
               frameon=False, fontsize=12, handlelength=5.0, handleheight=1.4, columnspacing=3.0,
               bbox_to_anchor=(0.5, 0.0))
    title = scale + (' — SMOKE ONLY' if args.smoke else '')
    if args.preliminary:
        count = sum(r['config'] == scale for r in recs)
        title += f' — PRELIMINARY ({count}/{args.expected_runs} runs complete)'
        fig.text(.5, .945, args.snapshot_label + '\nBars: available-seed means · Black points: individual seeds · n: completed seeds',
                 ha='center', fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, .925 if args.preliminary else .97])
    suffix = '_preliminary' if args.preliminary else ''
    out = os.path.join(FIGDIR, f'foundation_{scale}{suffix}.png')
    fig.savefig(out, dpi=140); fig.savefig(os.path.splitext(out)[0] + '.pdf'); plt.close(fig); print('SAVED', out)
