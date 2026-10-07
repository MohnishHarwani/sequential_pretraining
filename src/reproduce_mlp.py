"""Prepare data, train MLPs, save checkpoints/probes, and produce both paper layouts.

MLP: all three datasets, eight widths and seeds 0/1/2 (144 runs).
Use --records to redraw completed runs without training.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def generate():
    from run import load_config
    return load_config('curriculum').generate()


def render(records, output, fractions):
    from analyze_mlp_configs import analyze
    sys.path.insert(0, str(ROOT/'figures'))
    from mlp_plots import from_records, primacy, mechanism
    from safe_io import atomic_json
    fields = ('config', 'task', 'width', 'seed', 'ord')
    expected = {tuple(cfg[k] for k in fields) for _, cfg in generate()}
    found = [tuple(r[k] for k in fields) for r in records]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError('Expected every dataset/width/ordering with seeds 0/1/2; '
                         'incomplete or duplicate records cannot produce final paper figures')
    output = Path(output).resolve()
    data = from_records(records, fractions)
    report = analyze(records, output/'analysis', make_plots=False, fractions=fractions)
    primacy(data, output/'figures/mlp_primacy')
    mechanism(data, output/'figures/mlp_overlap', fractions)
    figures = [str(output/'figures'/f'{name}.{ext}')
               for name in ['mlp_primacy', 'mlp_overlap'] for ext in ['png', 'pdf']]
    report.update(figures=figures, plots=True)
    atomic_json(output/'analysis/outputs.json', report)
    with (output/'analysis/RESULTS.md').open('a') as stream:
        stream.write('\n[Primacy curves](../figures/mlp_primacy.png) · '
                     '[PDF](../figures/mlp_primacy.pdf)\n\n'
                     '[Representational overlap](../figures/mlp_overlap.png) · '
                     '[PDF](../figures/mlp_overlap.pdf)\n')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT/'data')
    ap.add_argument('--output', type=Path, default=ROOT/'results/mlp')
    ap.add_argument('--fractions', nargs='+', type=float, default=[.01, .05, .10],
                    help='Top-neuron fractions for overlap; saved scores support later changes')
    ap.add_argument('--records', type=Path, help='Only analyze this completed JSONL; no training or downloads')
    ap.add_argument('--dry-run', action='store_true', help='Show selection and destinations without executing')
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT/'figures'))
    from mlp_plots import validate_fractions
    fractions = validate_fractions(a.fractions)
    output = a.output.resolve()
    runs = generate()
    if a.dry_run:
        print(json.dumps(dict(runs=len(runs), seeds=[0, 1, 2],
                              mode='analysis' if a.records else 'train_and_analyze',
                              output=str(output), figures=str(output/'figures')), indent=2))
        return 0
    records_path = a.records.resolve() if a.records else output/'runs.jsonl'
    if not a.records:
        from run import execute
        execute(runs, 'mlp', a.data_dir, records_path)
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    print(json.dumps(render(records, output, fractions), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
