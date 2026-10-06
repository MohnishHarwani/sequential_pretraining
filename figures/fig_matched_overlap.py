"""Seed-mean MLP accuracy and training-probe overlap from analysis CSVs."""
import argparse
from pathlib import Path
from mlp_plots import load_tables, mechanism, PROBE_COMPARISONS


def plot(analysis, accuracy_summary, output, order='junkfirst', config='A',
         fractions=(.01, .05, .10), comparison='training'):
    if order != 'junkfirst':
        raise ValueError('The mechanism figure uses OOD-first overlap')
    configs = list('ABC') if config == 'all' else [config]
    data = load_tables(analysis, configs)
    name = 'mlp_ABC_training_overlap' if config == 'all' else 'mlp_training_overlap'
    if comparison != 'training':
        name = name.replace('training', comparison)
    mechanism(data, Path(output)/name, fractions, comparison)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--accuracy-summary', type=Path,
                        help='Accuracy summary in the analysis directory (optional)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', choices=['A', 'B', 'C', 'all'], default='A')
    parser.add_argument('--fractions', type=float, nargs='+', default=[.01, .05, .10])
    parser.add_argument('--comparison', choices=list(PROBE_COMPARISONS), default='training')
    args = parser.parse_args()
    if args.accuracy_summary is not None and args.config != 'all':
        if args.accuracy_summary.resolve() != (args.analysis/'accuracy_summary.csv').resolve():
            parser.error('--accuracy-summary must identify the analysis directory summary')
    plot(args.analysis, args.accuracy_summary, args.output, config=args.config,
         fractions=args.fractions, comparison=args.comparison)
