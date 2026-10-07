"""Automatic dataset preparation and checkpoint destinations for every trainer."""
import hashlib
import json
from pathlib import Path
import re
from safe_io import file_lock

ROOT = Path(__file__).resolve().parents[1]
_prepared = {}


def _signature(paths):
    """Invalidate the process-local preparation cache when an artifact changes."""
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) if p.is_file()
                 else (str(p), None, None) for p in paths)


def prepare_training(cfg, family):
    """Return a private config copy with ready data and a checkpoint directory.

    Called by the trainer API itself, so CLI, campaign, and Python callers share
    the same behavior. Existing prepared data are verified and reused.
    """
    cfg = dict(cfg)
    data = Path(cfg.get('data_dir') or ROOT / 'data').resolve()
    cfg['data_dir'] = str(data)
    data.mkdir(parents=True, exist_ok=True)
    if family == 'vision':
        from prepare_data import build_image
        task = cfg['task']
        paths = [data / f'{task}_{suffix}' for suffix in
                 ('x.npy', 'y.npy', 'layout.json', 'extra_validation.npz')]
        key = (family, str(data), task)
        with file_lock(data / f'.{task}.prepare.lock'):
            if _prepared.get(key) != _signature(paths):
                build_image(task, str(data))
                _prepared[key] = _signature(paths)
    elif family == 'foundation':
        from foundation_sources import SOURCES, prepare_corpus
        for name in dict.fromkeys((cfg['good_data'], cfg['ood_data'])):
            match = re.fullmatch(r'([a-z]+)_([1-9][0-9]*)MB\.npy', name)
            if not match or match[1] not in SOURCES:
                raise ValueError(f'Expected a public corpus filename <corpus>_<MB>MB.npy, got {name!r}')
            corpus, mb = match[1], int(match[2])
            paths = [data / name, (data / name).with_suffix('.split.json'),
                     data / f'{corpus}_validation.npy']
            key = (family, str(data), name)
            if _prepared.get(key) != _signature(paths):
                # prepare_corpus handles locking, interrupted preparation, and checksums.
                prepare_corpus(corpus, mb, data)
                _prepared[key] = _signature(paths)
    else:
        raise ValueError(f'Unknown trainer family: {family}')

    if not cfg.get('ckpt_dir'):
        operational = {'ckpt_dir', 'probe_out', 'log_every', 'progress', 'resume',
                       'checkpoint_every', 'stop_after'}
        identity = {k: v for k, v in cfg.items() if k not in operational}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cfg['ckpt_dir'] = str(ROOT / 'results' / 'standalone' / family / key)
    cfg['ckpt_dir'] = str(Path(cfg['ckpt_dir']).resolve())
    Path(cfg['ckpt_dir']).mkdir(parents=True, exist_ok=True)
    return cfg
