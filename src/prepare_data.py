"""Download and build every dataset the experiments use.

    python src/prepare_data.py --group vision            # image tasks (curriculum + mechanism)
    python src/prepare_data.py --group foundation --scale 100M    # or 500M / 1B / all
    python src/prepare_data.py --group all --scale all   # everything

    python src/prepare_data.py --corpus code --mb 500    # one foundation corpus at one size

Sources are fetched automatically from public Hugging Face datasets.

Image tasks  -> <task>_x.npy (float32 [N, features], raw pixel values) and <task>_y.npy (int64 [N])
Foundation corpora -> flat uint8 UTF-8 streams with shared held-out data

Foundation corpora are truncated to an exact byte budget and named `<corpus>_<MB>MB.npy`, so the
file name states its size. Budgets per scale:

    scale   target corpora (fineweb, tinystories)   OOD corpora (code, math, dewiki, fiwiki)
    100M    345 MB                                  500 MB
    500M    1035 MB                                 1500 MB
    1B      1665 MB                                 1500 MB
"""
import argparse, json, os, numpy as np
from pathlib import Path
from image_data import (flatten_images, additional_validation_count, load_extra_validation,
                        MLP_SPLIT_VERSION, SPLIT_SEED)
from foundation_sources import prepare_corpus

# ---------------------------------------------------------------- image tasks
# task -> (hub path, config, image column, label column)
IMAGES = {
    'mnist':      ('ylecun/mnist', None, 'image', 'label'),
    'fashion':    ('zalando-datasets/fashion_mnist', None, 'image', 'label'),
    'kmnist':     ('tanganke/kmnist', None, 'image', 'label'),
}

# ---------------------------------------------------------------- foundation corpora
SOURCES = {
    'fineweb':     ('HuggingFaceFW/fineweb', 'sample-10BT', ['text']),
    'tinystories': ('roneneldan/TinyStories', None, ['text']),
    'code':        ('sentence-transformers/codesearchnet', None, ['code', 'text']),
    'math':        ('open-web-math/open-web-math', None, ['text']),
    'dewiki':      ('wikimedia/wikipedia', '20231101.de', ['text']),
    'fiwiki':      ('wikimedia/wikipedia', '20231101.fi', ['text']),
}
TARGETS = ['fineweb', 'tinystories']
OOD = ['code', 'math', 'dewiki', 'fiwiki']
SCALES = {'100M': (345, 500), '500M': (1035, 1500), '1B': (1665, 1500)}


def build_image(task, out_dir, overwrite=False):
    """Keep the original training source and add test-split images for validation only."""
    from datasets import load_dataset
    xp = os.path.join(out_dir, f'{task}_x.npy')
    yp = os.path.join(out_dir, f'{task}_y.npy')
    layout_path = os.path.join(out_dir, f'{task}_layout.json')
    if os.path.exists(xp) and not overwrite:
        if os.path.exists(yp) and os.path.exists(layout_path):
            with open(layout_path) as f:
                if json.load(f).get('layout') == 'CHW':
                    print(f'{task}: original image data retained')
                    source_count = len(np.load(yp, mmap_mode='r'))
                    build_extra_validation(task, out_dir, source_count)
                    return
        raise ValueError(f'{task}: existing image layout is unverified; rebuild with --overwrite_images')
    path, cfg, xf, yf = IMAGES[task]
    ds = load_dataset(path, cfg, split='train') if cfg else load_dataset(path, split='train')
    x = flatten_images(np.stack([np.asarray(im, dtype=np.float32) for im in ds[xf]]), task)
    y = np.asarray(ds[yf], dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    np.save(xp, x); np.save(yp, y)
    with open(layout_path, 'w') as f:
        json.dump({'layout': 'CHW', 'task': task}, f)
    print(f'{task}: x{tuple(x.shape)} y{tuple(y.shape)}, {int(y.max()) + 1} classes')
    build_extra_validation(task, out_dir, len(y), overwrite=overwrite)


def build_extra_validation(task, out_dir, source_count, overwrite=False):
    from datasets import load_dataset
    output = Path(out_dir)/f'{task}_extra_validation.npz'
    if output.exists() and not overwrite:
        x, _, _, _ = load_extra_validation(task, out_dir, source_count)
        print(f'{task}: {len(x)} additional validation images already prepared')
        return
    path, cfg, xf, yf = IMAGES[task]
    dataset = load_dataset(path, cfg, split='test') if cfg else load_dataset(path, split='test')
    count = additional_validation_count(source_count)
    if len(dataset) < count:
        raise ValueError(f'{task}: test split has {len(dataset)} images; need {count}')
    indices = np.random.default_rng(SPLIT_SEED).permutation(len(dataset))[:count]
    selected = dataset.select(indices.tolist())
    x = flatten_images(np.stack([np.asarray(im, dtype=np.float32) for im in selected[xf]]), task)
    y = np.asarray(selected[yf], dtype=np.int64)
    meta = dict(version=MLP_SPLIT_VERSION, task=task, source=path, source_config=cfg,
                split='test', source_train_count=source_count, source_test_count=len(dataset),
                source_fingerprint=dataset._fingerprint, selection_seed=SPLIT_SEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output)+'.tmp', 'wb') as stream:
        np.savez_compressed(stream, x=x, y=y, test_indices=indices,
                            metadata_json=np.array(json.dumps(meta, sort_keys=True)))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(str(output)+'.tmp', output)
    print(f'{task}: added {count} official test images exclusively to validation')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', choices=['vision', 'foundation', 'all'], help='what to build')
    ap.add_argument('--scale', choices=list(SCALES) + ['all'], help='foundation scale(s)')
    ap.add_argument('--corpus', choices=list(SOURCES), help='build a single foundation corpus')
    ap.add_argument('--mb', type=int, help='byte budget in MB (required with --corpus)')
    ap.add_argument('--overwrite_images', action='store_true', help='rebuild existing image arrays in CHW order')
    ap.add_argument('--overwrite_foundation', action='store_true', help='rebuild legacy foundation data with shared validation')
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    a = ap.parse_args()

    if a.corpus:
        if not a.mb:
            ap.error('--corpus needs --mb')
        prepare_corpus(a.corpus, a.mb, a.out_dir, overwrite=a.overwrite_foundation)
        return
    if not a.group:
        ap.error('pass --group (vision / foundation / all) or --corpus')

    if a.group in ('vision', 'all'):
        for task in IMAGES:
            build_image(task, a.out_dir, overwrite=a.overwrite_images)
    if a.group in ('foundation', 'all'):
        if not a.scale:
            ap.error('--group foundation needs --scale')
        jobs = []
        for sc in (SCALES if a.scale == 'all' else [a.scale]):
            tgt_mb, ood_mb = SCALES[sc]
            jobs += [(c, tgt_mb) for c in TARGETS] + [(c, ood_mb) for c in OOD]
        for corpus, mb in sorted(set(jobs)):
            prepare_corpus(corpus, mb, a.out_dir, overwrite=a.overwrite_foundation)


if __name__ == '__main__':
    main()
