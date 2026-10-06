"""A/B/C rows and overlap-threshold columns, with three-seed mean curves."""
import argparse
from pathlib import Path
from mlp_plots import load_tables, mechanism, PROBE_COMPARISONS


def plot(analysis, output, fractions=(.01, .05, .10), comparison='training'):
    mechanism(load_tables(analysis, list('ABC')), output, fractions, comparison)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[.01, .05, .10])
    parser.add_argument('--comparison', choices=list(PROBE_COMPARISONS), default='training')
    args = parser.parse_args()
    plot(args.analysis, args.output, args.fractions, args.comparison)
