"""Shared MLP publication layouts: paired seed curves and matched-probe overlap."""
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mechanism_archive import overlap_scores, PROBE_COMPARISONS

DATASETS = [('mnist', 'MNIST'), ('fashion', 'Fashion-MNIST'), ('kmnist', 'KMNIST')]
COLORS = ['#0072B2', '#E69F00', '#009E73']
SETTINGS = {'A': 'LR 3e-4 / bs 256', 'B': 'LR 1e-3 / bs 64', 'C': 'LR 5e-4 / bs 32'}


def validate_fractions(fractions):
    fractions = tuple(fractions)
    if not fractions or any(not 0 < f <= 1 for f in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError('Provide distinct overlap fractions in (0, 1]')
    return fractions


def from_records(records, fractions=(.01, .05, .10), comparison='training', include_overlap=True):
    """Pair each seed's final accuracies; reject duplicates or incomplete pairs."""
    fractions = validate_fractions(fractions)
    groups = {}
    for r in records:
        if r.get('arch') != 'mlp':
            raise ValueError('Expected MLP records only')
        key = (r['config'], r['task'], r['width'], r['seed'])
        arms = groups.setdefault(key, {})
        if r['ord'] in arms:
            raise ValueError(f'Duplicate run: {key}/{r["ord"]}')
        arms[r['ord']] = r
    data = {}
    for (config, dataset, width, seed), arms in sorted(groups.items()):
        if set(arms) != {'goodfirst', 'junkfirst'}:
            raise ValueError(f'Missing ordering: {config}/{dataset}/{width}/{seed}')
        good, junk = arms['goodfirst'], arms['junkfirst']
        if good['params_M'] != junk['params_M'] or good.get('validation') != junk.get('validation'):
            raise ValueError('Paired models must share architecture and validation protocol')
        block = data.setdefault(config, {'paired': [], 'accuracy': {}, 'overlap': []})
        block['paired'].append(dict(dataset=dataset, width=width, seed=seed,
                                    params_M=good['params_M'],
                                    primacy_sensitivity=good['val_acc_full'] - junk['val_acc_full']))
        if include_overlap:
            for score in overlap_scores(junk['mechanism']['activations'], fractions, comparison):
                block['overlap'].append(dict(dataset=dataset, width=width, seed=seed,
                                            order='junkfirst', comparison=comparison, **score))
    for block in data.values():
        for key in {(r['dataset'], int(r['width'])) for r in block['paired']}:
            rows = [r for r in block['paired'] if (r['dataset'], int(r['width'])) == key]
            require_seeds(rows)
            block['accuracy'][key] = dict(params_M=rows[0]['params_M'],
                primacy_sensitivity_mean=float(np.mean([r['primacy_sensitivity'] for r in rows])))
    if not data:
        raise ValueError('No MLP results to plot')
    return data


def load_tables(folder, configs):
    data = {}
    for config in configs:
        root = Path(folder) / config
        if not root.is_dir() and len(configs) == 1:
            root = Path(folder)
        def read(name):
            with (root / name).open() as stream:
                return list(csv.DictReader(stream))
        data[config] = dict(paired=read('accuracy_by_seed.csv'),
                            accuracy={(r['dataset'], int(r['width'])): r
                                      for r in read('accuracy_summary.csv')},
                            overlap=read('overlap_by_seed.csv'))
    return data


def require_seeds(rows):
    if sorted(int(r['seed']) for r in rows) != [0, 1, 2]:
        raise ValueError('Each dataset/width/ordering must have seeds 0, 1, 2 exactly once')


def style(ax, mechanism=False):
    ticks = [1e4, 1e5, 1e6, 5e6] if mechanism else [2e4, 5e4, 2e5, 5e5, 1e6, 2e6, 5e6]
    def fmt(v, _):
        return (f'{v / 1e6:.1f}M' if mechanism else f'{v / 1e6:.0f}M') if v >= 1e6 else f'{v / 1e3:.0f}K'
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    if mechanism:
        ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, color='#DDDDDD', lw=.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def export(fig, output):
    """Save both formats after checking every data point is finite and visible."""
    fig.canvas.draw()
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_transform() != ax.transData:  # reference lines use blended transforms
                continue
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite figure data')
            if x.size and not (x.min() >= ax.get_xlim()[0] and x.max() <= ax.get_xlim()[1]
                               and y.min() >= ax.get_ylim()[0] and y.max() <= ax.get_ylim()[1]):
                raise ValueError('Figure limits clip data')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(output.with_suffix(ext), dpi=200, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)


def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.linewidth': .9, 'xtick.direction': 'out', 'ytick.direction': 'out'})


def primacy(data, output):
    """Configuration rows, dataset columns, separate paired-seed lines."""
    setup()
    configs = sorted(data)
    fig, axes = plt.subplots(len(configs), 3, figsize=(16.5, 4 * len(configs) + .7), squeeze=False)
    for i, config in enumerate(configs):
        for j, (dataset, label) in enumerate(DATASETS):
            ax = axes[i, j]
            rows = [r for r in data[config]['paired'] if r['dataset'] == dataset]
            widths = sorted({int(r['width']) for r in rows})
            if not widths:
                raise ValueError(f'Missing dataset: {config}/{dataset}')
            for width in widths:
                require_seeds([r for r in rows if int(r['width']) == width])
            for seed, color in enumerate(COLORS):
                selected = sorted((r for r in rows if int(r['seed']) == seed), key=lambda r: int(r['width']))
                ax.plot([float(r['params_M']) * 1e6 for r in selected],
                        [float(r['primacy_sensitivity']) for r in selected],
                        '-o', color=color, lw=2, ms=5, mec='white', mew=.7, label=f'seed {seed}')
            style(ax)
            xs = [float(r['params_M']) * 1e6 for r in rows]
            ax.set_xlim(min(xs) / 1.12, max(xs) * 1.05)
            ax.axhline(0, color='#D9D9D9', lw=1, zorder=0)
            if i == 0:
                ax.set_title(label, fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=.12, right=.99, bottom=.13 if len(configs) == 1 else .065,
                        top=.79 if len(configs) == 1 else .89, hspace=.25, wspace=.26)
    for i, config in enumerate(configs):
        box = axes[i, 0].get_position()
        fig.text(.055, (box.y0 + box.y1) / 2, f'config {config}\n{SETTINGS.get(config, "")}',
                 fontsize=12, fontweight='bold', ha='center', va='center', rotation=90)
    fig.text(.014, .48, 'target-first − OOD-first\nfinal accuracy', rotation=90,
             ha='center', va='center', fontsize=11)
    fig.suptitle('MLP', fontsize=16, fontweight='bold', y=.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, .92 if len(configs) == 1 else .955),
               ncol=3, frameon=False, fontsize=10)
    fig.supxlabel('parameters', fontsize=11, x=.555, y=.015)
    export(fig, output)


def mechanism(data, output, fractions=(.01, .05, .10), comparison='training'):
    """Configuration rows, threshold columns, seed-mean accuracy and overlap."""
    fractions = validate_fractions(fractions)
    setup()
    configs = sorted(data)
    fig, axes = plt.subplots(len(configs), len(fractions),
                             figsize=(5.5 * len(fractions) + .9, 4 * len(configs) + 1.4), squeeze=False)
    for i, config in enumerate(configs):
        block = data[config]
        for j, fraction in enumerate(fractions):
            ax = axes[i, j]; other = ax.twinx(); allx = []
            for color, (dataset, _) in zip(COLORS, DATASETS):
                widths = sorted(w for ds, w in block['accuracy'] if ds == dataset)
                if not widths:
                    raise ValueError(f'Missing dataset: {config}/{dataset}')
                xs, gaps, overlaps = [], [], []
                for width in widths:
                    accuracy = block['accuracy'][(dataset, width)]
                    rows = [r for r in block['overlap'] if r['dataset'] == dataset and
                            int(r['width']) == width and r['order'] == 'junkfirst' and
                            r['comparison'] == comparison and float(r['fraction']) == fraction]
                    require_seeds(rows)
                    paired = [r for r in block['paired'] if r['dataset'] == dataset and int(r['width']) == width]
                    require_seeds(paired)
                    gap = float(np.mean([float(r['primacy_sensitivity']) for r in paired]))
                    if not np.isclose(gap, float(accuracy['primacy_sensitivity_mean']), atol=1e-12):
                        raise ValueError('Accuracy summary differs from paired-seed values')
                    xs.append(float(accuracy['params_M']) * 1e6); gaps.append(gap)
                    overlaps.append(float(np.mean([float(r['overlap']) for r in rows])))
                ax.plot(xs, gaps, '-o', color=color, lw=2, ms=5, mec='white', mew=.7)
                other.plot(xs, overlaps, '--s', color=color, lw=1.7, ms=5, mec='white', mew=.7, alpha=.9)
                allx.extend(xs)
            style(ax, mechanism=True)
            ax.set_xlim(min(allx) / 1.25, max(allx) * 1.18)
            other.spines['top'].set_visible(False)
            other.spines['left'].set_visible(False)
            for a in (ax, other):
                low, high = a.get_ylim(); a.set_ylim(low, high + .06 * (high - low))
            if i == 0:
                ax.set_title(f'top {fraction:.0%}', fontsize=12, fontweight='bold', pad=8)
            if i == len(configs) - 1:
                ax.set_xlabel('parameters')
    single_col = len(fractions) == 1
    fig.subplots_adjust(left=.16 if single_col else .09, right=.86 if single_col else .93,
                        bottom=.22 if len(configs) == 1 else .10,
                        top=.77 if len(configs) == 1 else .88, hspace=.28, wspace=.32)
    for i, config in enumerate(configs):
        if len(configs) > 1:
            box = axes[i, 0].get_position()
            fig.text(.04, (box.y0 + box.y1) / 2, f'config {config}\n{SETTINGS.get(config, "")}',
                     rotation=90, ha='center', va='center', fontweight='bold', fontsize=11)
    fig.text(.014, .48, 'target-first − OOD-first accuracy gap', rotation=90, ha='center', va='center')
    fig.text(.985, .48, 'representational overlap', rotation=90, ha='center', va='center', color='#444')
    fig.suptitle('MLP', fontsize=14, fontweight='bold', y=.98)
    fig.legend(handles=[Line2D([], [], color=c, lw=2.4, label=label)
                        for c, (_, label) in zip(COLORS, DATASETS)],
               loc='upper center', bbox_to_anchor=(.5, .94), ncol=3, frameon=False, fontsize=10)
    label = {'training': 'training → training', 'validation': 'validation → validation',
             'cross_probe': 'training → validation'}[comparison]
    fig.legend(handles=[Line2D([], [], color='#555', lw=2, marker='o', label='accuracy gap (solid, left axis)'),
                        Line2D([], [], color='#555', lw=2, ls='--', marker='s',
                               label=f'{label} overlap (dashed, right axis)')],
               loc='lower center', bbox_to_anchor=(.5, .01), ncol=1 if single_col else 2,
               frameon=False, fontsize=10, handlelength=3)
    export(fig, output)
