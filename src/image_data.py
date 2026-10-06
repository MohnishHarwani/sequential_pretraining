"""Image layout shared by dataset preparation and the vision trainer."""
import json
from pathlib import Path
import numpy as np

IMG_SHAPE = {'mnist': (1, 28, 28), 'fashion': (1, 28, 28), 'kmnist': (1, 28, 28)}
MLP_SPLIT_VERSION = 'mlp-90-10-extended-v1'
SPLIT_SEED = 12345
ORIGINAL_VALIDATION_SIZE = 4000


def mlp_split_indices(source_count):
    """Preserve the original training/validation identities and their order."""
    if source_count <= ORIGINAL_VALIDATION_SIZE:
        raise ValueError('MLP source must contain more than 4,000 images')
    permutation = np.random.default_rng(SPLIT_SEED).permutation(source_count)
    return permutation[ORIGINAL_VALIDATION_SIZE:], permutation[:ORIGINAL_VALIDATION_SIZE]


def additional_validation_count(source_count):
    training_count = source_count - ORIGINAL_VALIDATION_SIZE
    count = round(training_count / 9) - ORIGINAL_VALIDATION_SIZE
    if count < 0:
        raise ValueError('Source is too small for 90/10 while retaining the original validation set')
    return count


def load_extra_validation(task, data_dir, source_count):
    path = Path(data_dir)/f'{task}_extra_validation.npz'
    if not path.exists():
        raise ValueError(f'{path}: missing additional validation data; run '
                         f'python src/prepare_data.py --group vision --out_dir {str(data_dir)!r}')
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata_json']))
        x, y, indices = saved['x'], saved['y'], saved['test_indices']
    count = additional_validation_count(source_count)
    if (meta.get('version') != MLP_SPLIT_VERSION or meta.get('task') != task
            or meta.get('source_train_count') != source_count or meta.get('split') != 'test'
            or meta.get('selection_seed') != SPLIT_SEED or len(x) != count
            or y.shape != (count,) or indices.shape != (count,)
            or len(np.unique(indices)) != count):
        raise ValueError(f'{path}: invalid additional validation metadata or counts; rebuild image data')
    expected = np.random.default_rng(SPLIT_SEED).permutation(meta['source_test_count'])[:count]
    if not np.array_equal(indices, expected):
        raise ValueError(f'{path}: additional validation identities differ from the fixed selection')
    return flatten_images(x, task), y.astype(np.int64), indices, meta


def flatten_images(images, task):
    """Return float32 rows flattened in CHW order, accepting NHWC/NCHW/NHW.

    Already-flat arrays must already use CHW order: their previous layout cannot
    be inferred from shape alone.
    """
    x = np.asarray(images, dtype=np.float32)
    c, h, w = IMG_SHAPE[task]
    if x.ndim == 2 and x.shape[1] == c * h * w:
        return x
    if x.ndim == 3 and c == 1 and x.shape[1:] == (h, w):
        x = x[:, None, :, :]
    elif x.ndim == 4 and x.shape[1:] == (h, w, c):
        x = x.transpose(0, 3, 1, 2)
    if x.ndim != 4 or x.shape[1:] != (c, h, w):
        raise ValueError(f'{task}: unexpected image shape {x.shape}; expected CHW {(c, h, w)}')
    return np.ascontiguousarray(x).reshape(len(x), -1)
