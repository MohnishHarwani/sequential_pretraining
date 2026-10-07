"""One-command public pipeline: download data, train/resume, verify, and draw Figure 6.

python src/reproduce_foundation.py --scales 100M 500M 1B
Select any subset, or --scales all. Each size includes both orderings and Exposure
Therapy with seeds 0/1/2. Figures go to OUTPUT/figures; checkpoints to OUTPUT/SCALE/runs.
Run training on a GPU. Separate stages support cluster deployment without changing
scientific code: --stage prepare|init|worker|analyze. Default stage is all.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from campaign_queue import WorkQueue
from safe_io import atomic_json, file_lock
ROOT = Path(__file__).resolve().parents[1]


def generate(scale, source_root=None):
    spec = importlib.util.spec_from_file_location('foundation_config', (source_root or ROOT)/'configs/foundation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.generate(scale=scale)


def initialize(root, data_dir, scale='100M', microbatch=8, smoke=False):
    root = Path(root).resolve()
    with file_lock(root/'.initialize.lock'):
        return _initialize(root, data_dir, scale, microbatch, smoke)


def _initialize(root, data_dir, scale, microbatch, smoke):
    from foundation_data import load_corpus
    root = Path(root).resolve(); data_dir = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'
    settings = dict(scale=scale, microbatch=microbatch, smoke=smoke, data_dir=str(data_dir))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['settings'] != settings:
            raise ValueError('Existing campaign settings differ; use a new output directory')
        WorkQueue(root)  # Also recover an interruption immediately after manifest publication.
        return manifest
    # The manifest is the commit marker. Before it exists, an unfinished snapshot
    # can be rebuilt without changing any committed campaign or training results.
    staging = root/'.code.staging'
    if staging.exists():
        shutil.rmtree(staging)
    for folder in ['src', 'configs', 'figures']:
        shutil.copytree(ROOT/folder, staging/folder, ignore=shutil.ignore_patterns('__pycache__'))
    configs = generate(scale, source_root=staging)
    if smoke:
        configs = configs[:3]  # all arms for one corpus pair, actual model and batch size
    data = {}
    for _, cfg in configs:
        for key in ['good_data', 'ood_data']:
            filename = cfg[key]
            if filename not in data:
                arr, _, meta = load_corpus(data_dir/filename)
                if hashlib.sha256(arr).hexdigest() != meta['training_sha256']:
                    raise ValueError(f'{filename}: training checksum mismatch')
                data[filename] = meta
    if (root/'code').exists():
        # Preserve old partial snapshots rather than deleting uncommitted files.
        (root/'code').rename(root/f'code.incomplete.{uuid.uuid4().hex}')
    staging.rename(root/'code')
    hashes = {str(p.relative_to(root/'code')): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root/'code').rglob('*.py')}
    tasks = []
    for driver, original in configs:
        cfg = dict(original)
        if smoke:
            cfg.update(phase1=20, phase2=20)
        good = cfg['good_data'].split('_')[0]; ood = cfg['ood_data'].split('_')[0]
        key = f"{scale}_{good}_{ood}_{cfg['ord']}_et{cfg['exposure_rate']:g}_s{cfg['seed']}"
        cfg.update(data_dir=str(data_dir), ckpt_dir=str(root/'runs'/key), microbatch=microbatch,
                   run_id=key, checkpoint_every=10 if smoke else 1000,
                   log_every=10 if smoke else 500, progress=True, resume=True)
        tasks.append(dict(id=key, driver=driver, cfg=cfg))
    manifest = dict(settings=settings, tasks=tasks, data=data, code_sha256=hashes,
                    smoke=smoke, scale=scale, created=time.time())
    import torch, numpy
    atomic_json(root/'environment.json', dict(python=sys.version, executable=sys.executable,
                torch=torch.__version__, cuda=torch.version.cuda, numpy=numpy.__version__))
    atomic_json(manifest_path, manifest)
    WorkQueue(root)
    return manifest


def worker(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    for path, digest in manifest['code_sha256'].items():
        if hashlib.sha256((root/'code'/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source changed: {path}')
    frozen_script = root/'code/src/reproduce_foundation.py'
    if str(frozen_script.relative_to(root/'code')) not in manifest['code_sha256']:
        raise ValueError('Campaign manifest does not identify its frozen worker script')
    if Path(__file__).resolve() != frozen_script.resolve():
        # A fresh interpreter also prevents previously imported live modules from
        # leaking into a campaign launched through the public Python API.
        return subprocess.run([sys.executable, str(frozen_script), '--stage', 'worker',
                               '--output', str(root)]).returncode
    import torch
    import runtime
    from runtime import TrainingInterrupted
    import train_foundation
    for module in [runtime, train_foundation]:
        if Path(module.__file__).resolve() != root/'code/src'/f'{module.__name__}.py':
            raise ValueError(f'Worker imported code outside its frozen snapshot: {module.__name__}')
    q = WorkQueue(root)
    if not torch.cuda.is_available():
        raise RuntimeError('Foundation training requires a GPU; data preparation and analysis use CPU')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    stopping = [False]
    def stop(sig, frame): stopping[0] = True
    signal.signal(signal.SIGUSR1, stop); signal.signal(signal.SIGTERM, stop)
    print(json.dumps(dict(event='worker_started', gpu=torch.cuda.get_device_name(),
                          account=os.environ.get('SLURM_JOB_ACCOUNT'), qos=os.environ.get('SLURM_JOB_QOS'))), flush=True)
    while not stopping[0]:
        # A failed task pauses the campaign instead of silently retrying bad training.
        if any(json.loads(p.read_text()) for p in (q.root/'failures').glob('*.json')):
            raise RuntimeError('Campaign contains a training failure; inspect failures/')
        claimed = q.claim()
        if claimed is None:
            if q.status()['completed'] == len(q.tasks):
                return 0
            time.sleep(10); continue
        task, lock, owner = claimed
        started = time.monotonic()
        print(json.dumps(dict(event='claimed', **owner)), flush=True)
        try:
            rec = train_foundation.run(dict(task['cfg']))
            assert Path(rec['checkpoint']).is_file()
            rec.update(run_id=task['id'], execution=owner, elapsed_seconds=time.monotonic()-started)
            q.complete(task, rec)
            print(json.dumps(dict(event='completed', task=task['id'], final=rec['final'])), flush=True)
        except TrainingInterrupted:
            print(json.dumps(dict(event='released_for_resume', task=task['id'])), flush=True)
            return 75
        except Exception:
            import traceback
            q.fail(task, traceback.format_exc())
            raise
        finally:
            lock.close()
    return 75


def render(records, scales, output, smoke=False):
    """Redraw saved results without downloading data or starting training."""
    records = Path(records).resolve()
    for scale in scales:
        args = [sys.executable, str(ROOT/'figures/fig_foundation.py'), '--scale', scale,
                '--input', str(records), '--out-dir', str(Path(output).resolve())]
        if smoke:
            args.append('--smoke')
        subprocess.run(args, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scales', '--scale', nargs='+', choices=['100M','500M','1B','all'],
                    help='Any subset, or all (default: 100M)')
    ap.add_argument('--data-dir', default=str(ROOT/'data/foundation'))
    ap.add_argument('--output', default=str(ROOT/'results/foundation'))
    ap.add_argument('--stage', choices=['all','prepare','init','worker','analyze'], default='all')
    ap.add_argument('--microbatch', type=int, default=8, help='Accumulation microbatch; optimizer batch stays 32')
    ap.add_argument('--smoke', action='store_true', help='Three 40-step runs at actual model scale; separate output required')
    ap.add_argument('--records', type=Path, help='Only draw this completed JSONL; no training or downloads')
    ap.add_argument('--dry-run', action='store_true', help='Show selected runs and destinations without executing')
    a = ap.parse_args(argv); output = Path(a.output).resolve()
    if a.microbatch < 1 or 32 % a.microbatch:
        ap.error('--microbatch must be a positive divisor of the optimizer batch size 32')
    direct = (output/'manifest.json').is_file()
    saved = json.loads((output/'manifest.json').read_text()) if direct else None
    selection = a.scales or ([saved['scale']] if direct else ['100M'])
    if len(set(selection)) != len(selection) or ('all' in selection and len(selection) != 1):
        ap.error('Use distinct model sizes, or all by itself')
    scales = ['100M','500M','1B'] if selection == ['all'] else selection
    if direct and scales != [saved['scale']]:
        ap.error('This output is an existing single-scale campaign; choose its scale or a new output')
    if a.records and a.stage != 'all':
        ap.error('--records draws saved results and cannot be combined with --stage')
    if a.stage == 'worker' and not direct:
        ap.error('--stage worker needs --output pointing to an initialized SCALE campaign directory')
    roots = {scale: output if direct else output/scale for scale in scales}
    if a.dry_run:
        print(json.dumps(dict(scales=scales, runs_per_scale=3 if a.smoke else 72,
                              runs=(3 if a.smoke else 72)*len(scales), seeds=[0] if a.smoke else [0,1,2],
                              mode='plot' if a.records else a.stage,
                              campaigns={s:str(p) for s,p in roots.items()},
                              figures=str(output/'figures')), indent=2))
        return 0
    if a.records:
        render(a.records, scales, output/'figures', a.smoke)
        return 0
    if a.stage == 'worker':
        return worker(output)
    for scale, root in roots.items():
        if a.stage in ['all','prepare']:
            from foundation_sources import prepare_scale
            prepare_scale(scale, a.data_dir)
            if a.stage == 'prepare': continue
        if a.stage in ['all','init']:
            m = initialize(root, a.data_dir, scale, a.microbatch, a.smoke)
            print(json.dumps(dict(campaign=str(root), tasks=len(m['tasks']), smoke=a.smoke)), flush=True)
            if a.stage == 'init': continue
        if a.stage == 'all':
            # Execute frozen code; reuse completed records and optimizer checkpoints.
            result = worker(root)
            if result: return result
        from analyze_foundation import analyze
        print(json.dumps(analyze(root), indent=2))
        manifest = json.loads((root/'manifest.json').read_text())
        render(root/'analysis/foundation.jsonl', [scale], output/'figures', manifest['smoke'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
