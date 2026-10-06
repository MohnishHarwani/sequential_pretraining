"""Compare matched validation/training probes using completed MLP campaigns only.

python src/analyze_probe_overlap.py CAMPAIGN [CAMPAIGN ...] --output OUTPUT
No training, downloads, or GPU execution. Existing campaign artifacts are read only.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from mechanism_archive import overlap_scores, PROBE_COMPARISONS

LABELS = {'validation': 'Validation at both phases', 'training': 'Training A at both phases',
          'cross_probe': 'Original: training A → validation'}
COLORS = {'validation': '#0072B2', 'training': '#D55E00', 'cross_probe': '#777777'}
DATASETS = ['mnist', 'fashion', 'kmnist']
TITLES = {'mnist': 'MNIST', 'fashion': 'Fashion-MNIST', 'kmnist': 'KMNIST'}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def draw_panel(ax, rows):
    for comparison in ['validation', 'training', 'cross_probe']:
        selected = sorted((r for r in rows if r['comparison'] == comparison), key=lambda r: r['width'])
        x = [r['width'] for r in selected]
        y = np.array([r['mean'] for r in selected]) * 100
        sd = np.array([r['std'] for r in selected]) * 100
        ax.plot(x, y, marker='o', ms=4, color=COLORS[comparison],
                ls='--' if comparison == 'cross_probe' else '-', label=LABELS[comparison], lw=1.8)
        ax.fill_between(x, np.maximum(0, y-sd), np.minimum(100, y+sd), color=COLORS[comparison], alpha=.08)
    ax.set_xscale('log', base=2); ax.set_ylim(0, 103)
    ax.set_xticks([8, 32, 128, 512, 2048], ['8', '32', '128', '512', '2048'])
    ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)


def figures(summary, output, fractions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})
    campaigns = sorted({r['campaign'] for r in summary})
    for order, order_label in [('junkfirst', 'OOD-first'), ('goodfirst', 'Target-first')]:
        for campaign in campaigns:
            fig, axes = plt.subplots(3, len(fractions), figsize=(5*len(fractions), 10), squeeze=False)
            for row, task in enumerate(DATASETS):
                for col, fraction in enumerate(fractions):
                    ax = axes[row, col]
                    selected = [r for r in summary if (r['campaign'], r['order'], r['dataset'], r['fraction'])
                                == (campaign, order, task, fraction)]
                    draw_panel(ax, selected)
                    ax.set_title(f'{TITLES[task]} · top {fraction:g}' if fraction*100 % 1 else
                                 f'{TITLES[task]} · top {fraction:.0%}')
                    if col == 0: ax.set_ylabel('Jaccard overlap (%)')
                    if row == 2: ax.set_xlabel('Hidden-layer width')
            fig.suptitle(f'MLP config A · {campaign} · {order_label}\n'
                         'Same probe at Phase 1 and final checkpoint · mean ± sample SD, 3 seeds', y=.995)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .947), ncol=3, frameon=False)
            fig.tight_layout(rect=[0, 0, 1, .90])
            for ext in ['png', 'pdf']:
                fig.savefig(output/f'overlap_{campaign}_{order}.{ext}', dpi=180)
            plt.close(fig)
        if .05 in fractions:
            fig, axes = plt.subplots(len(campaigns), 3, figsize=(15, 4*len(campaigns)), squeeze=False)
            for row, campaign in enumerate(campaigns):
                for col, task in enumerate(DATASETS):
                    ax = axes[row, col]
                    draw_panel(ax, [r for r in summary if (r['campaign'], r['order'], r['dataset'], r['fraction'])
                                   == (campaign, order, task, .05)])
                    ax.set_title(f'{TITLES[task]} · {campaign}')
                    if col == 0: ax.set_ylabel('Jaccard overlap (%)')
                    if row == len(campaigns)-1: ax.set_xlabel('Hidden-layer width')
            fig.suptitle(f'MLP config A · {order_label} · top 5% of units\n'
                         'Mean ± sample SD across 3 seeds; shaded bands clipped to 0–100%', y=.995)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .925), ncol=3, frameon=False)
            fig.tight_layout(rect=[0, 0, 1, .865])
            for ext in ['png', 'pdf']: fig.savefig(output/f'comparison_top5_{order}.{ext}', dpi=180)
            plt.close(fig)


def analyze(campaigns, output, fractions):
    output.mkdir(parents=True, exist_ok=False)
    rows, sources, records = [], [], []
    for campaign in campaigns:
        manifest = json.loads((campaign/'manifest.json').read_text())
        if manifest.get('smoke'): raise ValueError('Use completed full campaigns, not smoke runs')
        for task in manifest['tasks']:
            r = json.loads((campaign/'records'/f"{task['id']}.json").read_text())
            if r['arch'] != 'mlp' or r['config'] != 'A': raise ValueError('Expected config A MLP records')
            for key, path in r['mechanism'].items():
                if not Path(path).is_file(): raise ValueError(f'Missing {key}: {path}')
            path = Path(r['mechanism']['activations'])
            with np.load(path) as archive:
                for prefix in ['0X', 'TX', '0J', 'TJ']:
                    for layer in range(r['depth']):
                        if archive[f'{prefix}_{layer}'].shape != (r['width'],):
                            raise ValueError(f'{path}: incomplete activation vectors')
            original = overlap_scores(path)
            if r.get('overlap') != original:
                raise ValueError(f'{path}: original scores do not reproduce')
            with np.load(r['mechanism']['probe_context']) as ctx:
                if len(ctx['target_probe']) != 512 or len(ctx['junk_probe']) != 512:
                    raise ValueError('Expected saved 512-image probes')
                if not np.isin(ctx['target_source_indices'], ctx['validation_indices']).all():
                    raise ValueError('Target probe is not validation data')
                if not np.isin(ctx['junk_source_indices'], ctx['training_indices']).all():
                    raise ValueError('Corruption probe is not training data')
            sources.append(dict(run_id=r['run_id'], campaign=str(campaign), archive=str(path),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            records.append((campaign.name, r))
            for comparison in PROBE_COMPARISONS:
                for score in overlap_scores(path, fractions, comparison):
                    rows.append(dict(campaign=campaign.name, config=r['config'],
                                     dataset=r['task'], width=r['width'], seed=r['seed'], order=r['ord'],
                                     comparison=comparison, fraction=score['fraction'], overlap=score['overlap'],
                                     **{f'layer_{i}': x for i, x in enumerate(score['per_layer'])}))
        print(f'Analyzed {campaign.name}: {len(manifest["tasks"])} runs', flush=True)
    grouped = defaultdict(list)
    fields = ['campaign', 'config', 'dataset', 'width', 'order', 'comparison', 'fraction']
    for row in rows: grouped[tuple(row[k] for k in fields)].append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        if sorted(r['seed'] for r in group) != [0, 1, 2]: raise ValueError(f'Incomplete seeds: {key}')
        values = [r['overlap'] for r in group]
        summary.append(dict(zip(fields, key), seeds=3, mean=float(np.mean(values)), std=float(np.std(values, ddof=1))))
    write_csv(output/'overlap_by_seed.csv', rows); write_csv(output/'overlap_summary.csv', summary)
    (output/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    # Spot-check the widest OOD-first model for every dataset/campaign combination.
    import torch
    from compute_overlap import recompute_activations
    torch.set_num_threads(1)
    checks = []
    for campaign_name, r in records:
        if r['width'] != 2048 or r['seed'] != 0 or r['ord'] != 'junkfirst': continue
        dest = output/f"reconstructed_{campaign_name}_{r['task']}.npz"
        recompute_activations(r['mechanism'], dest, 'cpu')
        with np.load(dest) as rebuilt, np.load(r['mechanism']['activations']) as saved:
            errors = {}
            for key in rebuilt.files:
                np.testing.assert_allclose(rebuilt[key], saved[key], rtol=1e-4, atol=1e-5)
                errors[key] = float(np.max(np.abs(rebuilt[key]-saved[key])))
        checks.append(dict(run_id=r['run_id'], campaign=campaign_name, max_absolute_errors=errors))
        print(f'Checkpoint reconstruction passed: {campaign_name}, {r["task"]}, W2048', flush=True)
    figures(summary, output, fractions)
    metadata = dict(runs=len(records), comparisons=PROBE_COMPARISONS, fractions=fractions,
                    probe_size=512, original_scores_reproduced=True, checkpoint_spot_checks=checks,
                    training_performed=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'verification.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = ['# Matched-probe overlap from completed MLP config A runs', '',
             f'{len(records)} completed runs; three seeds; both orderings; no retraining.', '',
             'Validation compares 0X → TX. Training compares 0J → TJ on corruption-training set A.',
             'Original cross-probe comparison is 0J → TX. Scores are layer-mean Jaccard; uncertainty is sample SD.',
             'All use the original saved 512-image probes. The later validation expansion is not applied.',
             'Training here means the corruption-training probe, not all 56,000 training images.',
             'Full per-seed, per-layer values for both orderings are in overlap_by_seed.csv.',
             f'{len(checks)} widest-model checkpoint reconstructions match saved activations.', '',
             'At small widths, max(1, round(width × fraction)) makes some thresholds identical.', '',
             '## OOD-first, width 2048, top 5%', '',
             '| Campaign | Dataset | Validation → validation | Training A → training A | Original training A → validation |',
             '|---|---|---:|---:|---:|']
    for campaign in sorted({r['campaign'] for r in summary}):
        for task in DATASETS:
            cell = {}
            for r in summary:
                if (r['campaign'],r['dataset'],r['width'],r['order'],r['fraction']) == (campaign,task,2048,'junkfirst',.05):
                    cell[r['comparison']] = f"{100*r['mean']:.2f} ± {100*r['std']:.2f}%"
            if cell: lines.append(f"| {campaign} | {TITLES[task]} | {cell['validation']} | {cell['training']} | {cell['cross_probe']} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('campaigns', nargs='+', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--fractions', nargs='+', type=float, default=[.01, .05, .10])
    args = ap.parse_args()
    analyze([p.resolve() for p in args.campaigns], args.output.resolve(), args.fractions)
