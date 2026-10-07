"""Compute MLP overlap from saved snapshots, optionally rebuilding them from weights."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from mechanism_archive import atomic_npz, overlap_scores


@torch.no_grad()
def recompute_activations(paths, output, device='cpu'):
    from train_vision import MLP
    context = np.load(paths['probe_context'])
    snapshots = {}
    for phase, key, source in [('0J', 'phase1_checkpoint', 'junk_probe'),
                               ('TX', 'final_checkpoint', 'target_probe')]:
        payload = torch.load(paths[key], map_location=device, weights_only=False)
        weights, meta = payload['model'], payload['meta']
        net = MLP(weights['inp.weight'].shape[1], weights['hg.weight'].shape[0],
                  meta['width'], meta['depth']).to(device)
        net.load_state_dict(weights)
        net.eval()
        _, acts = net.trunk_acts(torch.tensor(context[source], device=device))
        for layer, act in enumerate(acts):
            snapshots[f'{phase}_{layer}'] = act.abs().mean(0).cpu().numpy()
    atomic_npz(output, snapshots)
    return overlap_scores(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('record', help='one completed-run JSON record')
    ap.add_argument('--recompute', help='recompute activations from phase weights into this NPZ')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--fractions', type=float, nargs='+', default=[.01, .05, .10])
    a = ap.parse_args()
    record = json.loads(Path(a.record).read_text())
    if record.get('arch') != 'mlp':
        raise ValueError('Overlap is defined only for MLP records')
    if record.get('ord') != 'junkfirst':
        raise ValueError('The paper overlap requires an OOD-first run')
    paths = record['mechanism']
    if a.recompute:
        recompute_activations(paths, a.recompute, a.device)
    scores = overlap_scores(a.recompute or paths['activations'], a.fractions)
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
