"""Persistent MLP phase checkpoints and inputs for post-hoc overlap analysis."""
import json
from pathlib import Path
import os
import numpy as np
import torch


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.tmp', 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(str(path) + '.tmp', path)


class MechanismArchive:
    def __init__(self, cfg, tag, final_checkpoint, probe_dir, inputs):
        if cfg.get('arch') != 'mlp':
            raise ValueError('Overlap mechanism archives are MLP-only')
        self.cfg = dict(cfg)
        self.final_checkpoint = final_checkpoint
        self.phase1_checkpoint = (str(Path(final_checkpoint).with_suffix('.phase1.pt'))
                                  if final_checkpoint else None)
        self.activations = str(Path(probe_dir) / (
            f"{cfg['task']}_W{cfg['width']}_{cfg['ord']}_s{cfg['seed']}_{cfg['config']}.npz"))
        self.context = str(Path(final_checkpoint).with_suffix('.mechanism.npz') if final_checkpoint else
                           Path(probe_dir) / (tag + '.mechanism.npz'))
        meta = dict(version=1, config=self.cfg, probe_size=len(inputs['junk_probe']),
                    unit_score='mean absolute post-ReLU activation', split_seed=12345)
        layout = Path(cfg['data_dir']) / (cfg['task'] + '_layout.json')
        if layout.exists():
            meta['dataset'] = json.loads(layout.read_text())
        atomic_npz(self.context, dict(inputs, metadata_json=np.array(json.dumps(meta, sort_keys=True))))

    def paths(self):
        return dict(phase1_checkpoint=self.phase1_checkpoint, final_checkpoint=self.final_checkpoint,
                    probe_context=self.context, activations=self.activations)

    def save_phase1(self, net, step):
        if self.phase1_checkpoint:
            path = self.phase1_checkpoint
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=net.state_dict(), step=step, meta=self.cfg,
                            mechanism=self.paths()), path + '.tmp')
            os.replace(path + '.tmp', path)

    def save_activations(self, snap):
        arrays = {f'{phase}_{layer}': values.astype(np.float32)
                  for phase, layers in snap.items() for layer, values in enumerate(layers)}
        arrays['metadata_json'] = np.array(json.dumps(dict(config=self.cfg, paths=self.paths()), sort_keys=True))
        atomic_npz(self.activations, arrays)


PROBE_COMPARISONS = {'cross_probe': ('0J', 'TX'),
                     'validation': ('0X', 'TX'), 'training': ('0J', 'TJ')}


def overlap_scores(path, fractions=(.01, .05, .10), comparison='training'):
    """Per-layer top-f Jaccard and its layer mean, matching the paper's metric."""
    first, last = PROBE_COMPARISONS[comparison]
    fractions = tuple(fractions)
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ValueError('Overlap fractions must be greater than zero and at most one')
    with np.load(path) as data:
        layers = sorted(int(k.split('_')[1]) for k in data.files if k.startswith(first + '_'))
        if not layers or any(f'{last}_{i}' not in data for i in layers):
            raise ValueError(f'{path}: both phase snapshots are required')
        result = []
        for fraction in fractions:
            scores = []
            for layer in layers:
                a, b = data[f'{first}_{layer}'], data[f'{last}_{layer}']
                if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f'{path}: invalid activation scores in layer {layer}')
                k = max(1, int(round(len(a) * fraction)))
                aa, bb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
                scores.append(len(aa & bb) / len(aa | bb))
            result.append(dict(fraction=fraction, per_layer=scores, overlap=float(np.mean(scores))))
        return result
