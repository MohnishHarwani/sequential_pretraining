"""Foundation corpus splits shared across model scales.

Documents are assigned to training/held-out streams by content hash before any
byte budget is applied. Every scale excludes all held-out documents, including
duplicates, and uses the same per-corpus validation file.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from safe_io import atomic_json

SPLIT_VERSION = 'sha256-document-v1'
VALIDATION_POOL_BYTES = 1_000_000


def is_held_out(text):
    digest = hashlib.sha256(text.encode('utf-8', 'ignore')).digest()
    return int.from_bytes(digest[:8], 'big') % 20 == 0


def recover_corpus(path):
    """Finish an interrupted array/metadata publication under the preparation lock."""
    path = Path(path)
    pending = path.with_suffix('.split.pending.json')
    if not pending.exists():
        return
    meta = json.loads(pending.read_text())
    temporary = path.with_name(path.name + '.tmp')
    candidate = temporary if temporary.exists() else path
    train = np.load(candidate, mmap_mode='r')
    val = np.load(path.parent/meta['validation_file'], mmap_mode='r')
    if (meta.get('split') != SPLIT_VERSION or train.ndim != 1 or train.dtype != np.uint8
            or len(train) != meta['training_bytes']
            or hashlib.sha256(train).hexdigest() != meta['training_sha256']
            or val.ndim != 1 or val.dtype != np.uint8
            or hashlib.sha256(val).hexdigest() != meta['validation_sha256']):
        raise ValueError(f'{path}: interrupted corpus does not match its pending metadata')
    del train, val  # Release mmap handles before renaming on systems that require it.
    if candidate == temporary:
        os.replace(temporary, path)
    os.replace(pending, path.with_suffix('.split.json'))


def build_corpus(records, fields, name, mb, out_dir, source, overwrite=False):
    """Write a training-only byte budget and a fixed, disjoint held-out stream.

    The approximately 95/5 split is over document hashes, not a tail of each
    budget-sized file. The held-out stream is the same prefix at every scale.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / f'{name}_{mb}MB.npy'
    meta_path = train_path.with_suffix('.split.json')
    pending = train_path.with_suffix('.split.pending.json')
    val_path = root / f'{name}_validation.npy'
    recover_corpus(train_path)
    if train_path.exists() and meta_path.exists() and not overwrite:
        load_corpus(train_path)
        print(f'{train_path.name}: verified split metadata, skipping')
        return
    target = int(mb * 1_000_000)
    if target <= 0:
        raise ValueError('Training byte budget must be positive')
    tmp = train_path.with_name(train_path.name + '.tmp')
    trimmed = tmp.with_name(tmp.name + '.trimmed')
    train = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(target,))
    count = 0
    held = bytearray()
    train_hash = hashlib.sha256()
    try:
        for rec in records:
            text = next((rec.get(f) for f in fields if isinstance(rec.get(f), str) and rec[f]), None)
            if text is None:
                continue
            raw = text.encode('utf-8', 'ignore') + b'\n\n'
            if is_held_out(text):
                held.extend(raw[:max(0, VALIDATION_POOL_BYTES - len(held))])
            elif count < target:
                chunk = raw[:target - count]
                train[count:count + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
                count += len(chunk)
                train_hash.update(chunk)
            if count == target and len(held) == VALIDATION_POOL_BYTES:
                break
        if count == 0 or len(held) != VALIDATION_POOL_BYTES:
            raise ValueError(f'{name}: source exhausted with {count}/{target} training bytes '
                             f'and {len(held)}/{VALIDATION_POOL_BYTES} held-out bytes')
        val = np.frombuffer(held, dtype=np.uint8)
        if val_path.exists():
            if not np.array_equal(np.load(val_path), val):
                raise ValueError(f'{val_path}: source changed; use a new data directory for all scales')
        else:
            with open(str(val_path) + '.tmp', 'wb') as f:
                np.save(f, val)
            os.replace(str(val_path) + '.tmp', val_path)
        train.flush()
        if count < target:
            with trimmed.open('wb') as stream:
                np.save(stream, train[:count])
        del train
        if count < target:
            os.replace(trimmed, tmp)
        meta = dict(split=SPLIT_VERSION, corpus=name, source=source,
                    training_bytes=count, requested_training_bytes=target,
                    training_sha256=train_hash.hexdigest(),
                    validation_file=val_path.name,
                    validation_sha256=hashlib.sha256(held).hexdigest())
        # Publish a recovery record before either committed file changes. A retry
        # can finish the pair without downloading or rebuilding the corpus.
        atomic_json(pending, meta)
        os.replace(tmp, train_path)
        os.replace(pending, meta_path)
        print(f'{train_path.name}: {count / 1e6:g} MB training; shared {val_path.name}')
    finally:
        if trimmed.exists():
            trimmed.unlink()
        if tmp.exists() and not pending.exists():
            tmp.unlink()


def load_corpus(path):
    """Require explicit split provenance; never silently use legacy tail splits."""
    path = Path(path)
    meta_path = path.with_suffix('.split.json')
    if not meta_path.exists():
        raise ValueError(f'{path}: missing shared-validation metadata; rebuild foundation data '
                         'with prepare_data.py --group foundation --scale ... --overwrite_foundation')
    with meta_path.open() as f:
        meta = json.load(f)
    if meta.get('split') != SPLIT_VERSION:
        raise ValueError(f'{meta_path}: unsupported corpus split')
    train = np.load(path, mmap_mode='r')
    val = np.load(path.parent / meta['validation_file'], mmap_mode='r')
    if train.ndim != 1 or train.dtype != np.uint8 or len(train) != meta['training_bytes']:
        raise ValueError(f'{path}: training array does not match its split metadata')
    if val.ndim != 1 or val.dtype != np.uint8 or hashlib.sha256(val).hexdigest() != meta['validation_sha256']:
        raise ValueError(f'{path}: shared validation file does not match its split metadata')
    return train, val, meta


def exposure_count(n_sequences, fraction, rate):
    """Exposure size is independent of the exposure rate and does not shrink training."""
    if not 0 <= fraction <= 1 or not 0 <= rate <= 1:
        raise ValueError('Exposure pool fraction and rate must lie in [0, 1]')
    if rate > 0 and fraction == 0:
        raise ValueError('Positive exposure rate requires a positive exposure pool fraction')
    if n_sequences < 1:
        raise ValueError('Target training pool has no complete sequences')
    return max(1, round(fraction * n_sequences)) if fraction else 0
