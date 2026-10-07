"""Plot MLP accuracy gaps and cross-phase representational overlap from saved run records."""
import argparse
from pathlib import Path
from _common import load, RESULTS, FIGDIR
from mlp_plots import from_records, mechanism


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=Path(RESULTS)/'mechanism.jsonl')
    parser.add_argument('--fractions', type=float, nargs='+', default=[.01, .05, .10])
    parser.add_argument('--output', type=Path, default=Path(FIGDIR)/'mechanism')
    args = parser.parse_args()
    records = load(str(args.records.resolve()))
    records = [r for r in records if r['config'] == 'mlp']
    data = from_records(records, args.fractions)
    mechanism(data, args.output, args.fractions)
    print('SAVED', args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
