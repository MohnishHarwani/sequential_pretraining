"""Pinned public corpora, downloaded automatically; no private cache is required."""
import fcntl
import hashlib
import json
import time
from pathlib import Path
from foundation_data import build_corpus, load_corpus, recover_corpus

SOURCES = {
 'fineweb': dict(path='HuggingFaceFW/fineweb', revision='9bb295ddab0e05d785b879661af7260fed5140fc', prefix='sample/10BT/', fields=['text']),
 'tinystories': dict(path='roneneldan/TinyStories', revision='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64', prefix='data/train-', fields=['text']),
 'code': dict(path='sentence-transformers/codesearchnet', revision='079a958b01dc87cf07b66a68414c4b4196d889cc', prefix='pair/train-', fields=['code']),
 'math': dict(path='open-web-math/open-web-math', revision='fde8ef8de2300f5e778f56261843dab89f230815', prefix='data/train-', fields=['text']),
 'dewiki': dict(path='wikimedia/wikipedia', revision='b04c8d1ceb2f5cd4588862100d08de323dccfbaa', prefix='20231101.de/train-', fields=['text']),
 'fiwiki': dict(path='wikimedia/wikipedia', revision='b04c8d1ceb2f5cd4588862100d08de323dccfbaa', prefix='20231101.fi/train-', fields=['text']),
}


def retry(fn):
    for attempt in range(6):
        try:
            return fn()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** (attempt + 1)))


def prepare_corpus(name, mb, out_dir, overwrite=False):
    """Pin both revision and shard order, and verify cached output before reuse."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    spec = SOURCES[name]
    path = root / f'{name}_{mb}MB.npy'
    with (root / f'.{name}.prepare.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recover_corpus(path)
        if path.exists() and path.with_suffix('.split.json').exists() and not overwrite:
            train, _, meta = load_corpus(path)
            if any(meta['source'].get(k) != v for k, v in spec.items()):
                raise ValueError(f'{path}: source differs from pinned version; use a new data directory')
            if hashlib.sha256(train).hexdigest() != meta['training_sha256']:
                raise ValueError(f'{path}: training checksum mismatch')
            print(f'{path.name}: cached corpus verified', flush=True)
            return meta
        info = retry(lambda: HfApi().dataset_info(spec['path'], revision=spec['revision']))
        files = sorted(x.rfilename for x in info.siblings
                       if x.rfilename.startswith(spec['prefix']) and x.rfilename.endswith('.parquet'))
        if not files:
            raise ValueError(f'{name}: no matching parquet shards')
        source = dict(spec, split='train', shards=[])
        def records():
            for filename in files:
                print(f'{name}: downloading {filename}', flush=True)
                local = retry(lambda: hf_hub_download(spec['path'], filename, repo_type='dataset',
                              revision=spec['revision'], cache_dir=str(root / '.hub')))
                source['shards'].append(filename)
                pf = pq.ParquetFile(local)
                for batch in pf.iter_batches(batch_size=1024, columns=spec['fields']):
                    yield from batch.to_pylist()
        build_corpus(records(), spec['fields'], name, mb, root, source, overwrite=overwrite)
        return load_corpus(path)[2]


def prepare_scale(scale, out_dir):
    from prepare_data import SCALES, TARGETS, OOD
    target_mb, ood_mb = SCALES[scale]
    jobs = [(c, target_mb) for c in TARGETS] + [(c, ood_mb) for c in OOD]
    return {name: prepare_corpus(name, mb, out_dir) for name, mb in jobs}
