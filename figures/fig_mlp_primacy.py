"""Primacy curves: dataset panels, separate lines for seeds 0/1/2."""
import argparse
from pathlib import Path
from _common import load, RESULTS, FIGDIR
from mlp_plots import from_records, primacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=Path(RESULTS)/'curriculum.jsonl')
    parser.add_argument('--output', type=Path, default=Path(FIGDIR)/'mlp_primacy')
    args = parser.parse_args()
    records = load(str(args.records.resolve()))
    records = [r for r in records if r['config'] == 'mlp']
    primacy(from_records(records, include_overlap=False), args.output)
    print('SAVED', args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
