"""Seed-mean MLP accuracy and cross-phase representational overlap from analysis CSVs."""
import argparse
from pathlib import Path
from mlp_plots import load_tables, mechanism


def plot(analysis, accuracy_summary, output, order='junkfirst', config='mlp',
         fractions=(.01, .05, .10)):
    if order != 'junkfirst':
        raise ValueError('The mechanism figure uses OOD-first overlap')
    if config != 'mlp':
        raise ValueError('Expected the default MLP experiment')
    configs = ['mlp']
    data = load_tables(analysis, configs)
    mechanism(data, Path(output)/'mlp_overlap', fractions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--accuracy-summary', type=Path,
                        help='Accuracy summary in the analysis directory (optional)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[.01, .05, .10])
    args = parser.parse_args()
    if args.accuracy_summary is not None:
        if args.accuracy_summary.resolve() != (args.analysis/'accuracy_summary.csv').resolve():
            parser.error('--accuracy-summary must identify the analysis directory summary')
    plot(args.analysis, args.accuracy_summary, args.output,
         fractions=args.fractions)
