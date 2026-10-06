"""Run an experiment locally and collect its results.

Expands configs/<experiment>.py into a run list, executes each run with the right trainer,
and writes one JSON record per run to results/<experiment>.jsonl. MLP runs
also retain per-neuron activation probes alongside their checkpoints. The figure scripts
read only these result files -- no SLURM, no stdout scraping.

Usage:
    python src/run.py curriculum  --data_dir /path/to/data
    python src/run.py mechanism   --data_dir /path/to/data
    python src/run.py foundation  --data_dir /path/to/data     # A100 / 1B-scale compute

    python src/run.py curriculum --dry_run            # print the run list, execute nothing
    python src/run.py curriculum --limit 4            # first 4 full-length runs only

MLP datasets and additional validation images are downloaded/prepared automatically.
Completed runs are preserved and skipped on restart. Interrupted toy runs restart
individually; foundation runs resume their optimizer/RNG checkpoints. A changed
configuration requires a new --out file. Final weights live in per-run directories
under <out>.artifacts; --no_ckpt disables this, --ckpt_dir moves that artifact root.
"""
import os, sys, json, argparse, importlib.util, hashlib
from pathlib import Path
from safe_io import atomic_json, atomic_text, file_lock
from image_data import MLP_SPLIT_VERSION

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import train_vision, train_foundation
DRIVERS = {'vision': train_vision, 'foundation': train_foundation}


def load_config(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'configs', name + '.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def execute(runs, experiment, data_dir, out, ckpt_dir=None, no_ckpt=False, limit=None):
    """Commit complete records atomically; never truncate or mix configurations."""
    out = Path(out).resolve()
    # Even an explicitly shared artifact root is isolated by output identity.
    output_id = hashlib.sha256(str(out).encode()).hexdigest()[:16]
    artifacts = (Path(ckpt_dir).resolve() / output_id if ckpt_dir
                 else Path(str(out) + '.artifacts'))
    plan = []
    for driver, original in runs:
        cfg = dict(original, data_dir=str(Path(data_dir).resolve()))
        if driver == 'vision':
            cfg.setdefault('validation_protocol', MLP_SPLIT_VERSION)
        key = hashlib.sha256(json.dumps([driver, cfg], sort_keys=True).encode()).hexdigest()
        cfg.pop('ckpt_dir', None)
        cfg.pop('probe_out', None)
        if not no_ckpt:
            cfg['ckpt_dir'] = str(artifacts / key)
        if experiment == 'mechanism' and cfg.get('ord') == 'junkfirst':
            cfg['probe_out'] = str(artifacts / key / 'mechanism_probes')
        plan.append(dict(id=key, driver=driver, cfg=cfg))
    if len({p['id'] for p in plan}) != len(plan):
        raise ValueError('Configuration contains duplicate runs')
    manifest = dict(version=1, experiment=experiment, runs=plan)
    manifest_path = Path(str(out) + '.manifest.json')
    with file_lock(str(out) + '.lock'):
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('Existing run configuration differs; choose a new --out file')
        elif out.exists() and out.stat().st_size:
            raise ValueError('Existing results lack a configuration manifest; preserved unchanged. '
                             'Choose a new --out file for resumable execution')
        else:
            atomic_json(manifest_path, manifest)
        records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
        completed = set()
        expected = {p['id'] for p in plan}
        for record in records:
            key = record.get('runner_id')
            if key not in expected or key in completed:
                raise ValueError('Results do not match the run manifest; preserved unchanged')
            completed.add(key)
        selected = plan[:limit] if limit is not None else plan
        prepared = set()
        for i, task in enumerate(selected, 1):
            if task['id'] in completed:
                print(f'  [{i}/{len(selected)}] already complete; skipping')
                continue
            cfg = dict(task['cfg'])
            if task['driver'] == 'vision' and cfg['task'] not in prepared:
                from prepare_data import build_image
                build_image(cfg['task'], cfg['data_dir'])
                prepared.add(cfg['task'])
            record = DRIVERS[task['driver']].run(cfg)
            record['runner_id'] = task['id']
            # Replacing the complete JSONL avoids a torn last record on interruption.
            records.append(record)
            atomic_text(out, ''.join(json.dumps(r) + '\n' for r in records))
            completed.add(task['id'])
            print(f"  [{i}/{len(selected)}] {task['driver']} {cfg.get('task', cfg.get('good_data', ''))} done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiment', choices=['curriculum', 'mechanism', 'foundation', 'mlp_a'])
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None, help='run only the first N configurations; training length is unchanged')
    ap.add_argument('--ckpt_dir', default=None, help='where model weights are written '
                                                     '(default <out>.artifacts)')
    ap.add_argument('--no_ckpt', action='store_true', help='do not save model weights')
    ap.add_argument('--dry_run', action='store_true', help='print the run list and exit')
    a = ap.parse_args()

    runs = load_config(a.experiment).generate()
    if a.limit is not None and a.limit < 1:
        ap.error('--limit must be positive')
    out = a.out or os.path.join(ROOT, 'results', a.experiment + '.jsonl')
    selected = runs[:a.limit] if a.limit is not None else runs
    print(f"[{a.experiment}] {len(selected)} runs -> {out}")
    if a.dry_run:
        for driver, cfg in selected[:12]:
            print(' ', driver, {k: cfg[k] for k in ('task', 'width', 'ord', 'config') if k in cfg})
        print('  ...' if len(runs) > 12 else '')
        return

    execute(runs, a.experiment, a.data_dir, out, a.ckpt_dir, a.no_ckpt, a.limit)


if __name__ == '__main__':
    main()
