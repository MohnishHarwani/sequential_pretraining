"""Shared helpers for the figure scripts."""
import os, json, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get('REPRO_RESULTS', os.path.join(ROOT, 'results'))   # override with REPRO_RESULTS
FIGDIR = os.path.join(RESULTS, 'figures')
OKABE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9']   # colorblind-safe categorical


def load(name):
    path = name if os.path.isabs(name) else os.path.join(RESULTS, name)
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def final_accuracy(record):
    """Final full-held-out accuracy, consistently across toy architectures.

    Older MLP records evaluated the whole held-out set at their final logged
    step, so that point is a valid fallback when the explicit final score is absent.
    """
    if record.get('val_acc_full') is not None:
        return float(record['val_acc_full'])
    if record.get('arch') == 'mlp' and record.get('vacc'):
        return float(record['vacc'][-1])
    raise ValueError('Missing final full-held-out accuracy; re-score the saved checkpoint')


def top_frac_idx(v, f):
    k = max(1, int(round(len(v) * f)))
    return set(np.argsort(-v)[:k].tolist())


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))
