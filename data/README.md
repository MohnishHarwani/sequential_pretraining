# Datasets

This directory contains documentation in the source release. Dataset arrays are
downloaded and generated locally by the scripts, and are excluded by `.gitignore`.
No prebuilt arrays, private caches, or manually supplied files are required.

`python src/run.py mechanism` automatically downloads and prepares all required
MLP data before training. For data preparation alone, run:

```bash
python src/prepare_data.py --group vision
```

Use `--out_dir /path/to/data` for a different preparation destination, and pass
that directory to the runner with `--data_dir /path/to/data`.

## Image datasets (curriculum + mechanism)
Run `python src/prepare_data.py --group vision`. For each `<task>`, the builder writes
two arrays and `<task>_layout.json` recording CHW layout:

| file | shape | dtype | notes |
|------|-------|-------|-------|
| `<task>_x.npy` | `[N, C*H*W]` | float32 | raw pixels flattened in CHW order; the trainer applies training-set normalization |
| `<task>_y.npy` | `[N]` | int | class labels `0..C-1` |

Tasks used: `mnist`, `fashion`, `kmnist` (grayscale 28×28 images, 10 classes).
The image-shape table lives in `src/image_data.py` (`IMG_SHAPE`). The trainer accepts
flattened rows, grayscale NHW arrays, and NHWC/NCHW arrays with a singleton channel.
Rebuild prepared images with
`python src/prepare_data.py --group vision --overwrite_images`. Existing arrays without
layout metadata are not silently reused by the preparation script.

Preparation also writes `<task>_extra_validation.npz`, containing 2,222 additional
images and labels from the official test split, their source indices, and source
metadata. The original 60,000-image arrays remain unchanged. The trainer keeps
the same 56,000 training images and 4,000 original validation images (shuffle seed
12345), then appends the additional images to validation only. The resulting split
is **56,000 / 6,222**, approximately 90/10. Extra test images are selected without
replacement using seed 12345. The other 7,778 official test images are unused.

Run the same preparation command to extend existing datasets. Training requires
the additional-validation archive and will explain how to prepare it if missing.
MLP result records and checkpoints identify this validation protocol and its counts.
Mechanism archives retain the original validation probe prefix and record the added
images' official test indices separately; indices above 59,999 refer to appended
rows, not to original training-source rows.

## Foundation datasets (100M / 500M / 1B)
**These are built for you** — no manual preparation:

```bash
python src/prepare_data.py --group foundation --scale 100M      # or 500M / 1B / all
```

Training-only files are named `<corpus>_<MB>MB.npy`, so the name states their byte budget:

| scale | target corpora | OOD corpora |
|---|---|---|
| 100M | `fineweb_345MB.npy`, `tinystories_345MB.npy` | `code_500MB.npy`, `math_500MB.npy`, `dewiki_500MB.npy`, `fiwiki_500MB.npy` |
| 500M | `fineweb_1035MB.npy`, `tinystories_1035MB.npy` | `code_1500MB.npy`, `math_1500MB.npy`, `dewiki_1500MB.npy`, `fiwiki_1500MB.npy` |
| 1B | `fineweb_1665MB.npy`, `tinystories_1665MB.npy` | the same 1500 MB OOD files as 500M |

Sources (pinned Parquet shards are downloaded from the Hugging Face hub until
the training and validation budgets are filled; downloads can exceed those budgets):

| corpus | dataset | config |
|---|---|---|
| fineweb | `HuggingFaceFW/fineweb` | `sample-10BT` |
| tinystories | `roneneldan/TinyStories` | — |
| code | `sentence-transformers/codesearchnet` | — |
| math | `open-web-math/open-web-math` | — |
| dewiki | `wikimedia/wikipedia` | `20231101.de` |
| fiwiki | `wikimedia/wikipedia` | `20231101.fi` |

Each training file is a flat `uint8` stream of UTF-8 bytes with documents separated by a
blank line, truncated to the stated training budget. Before collecting bytes, a stable
SHA-256 hash assigns each document to training (approximately 95%) or held-out (5%).
Duplicate document text always receives the same assignment. No held-out document is
included in a training file at any model scale.

Each corpus also has a shared `<corpus>_validation.npy`: the first 1 MB of the held-out
stream, identical at every scale. The trainer samples approximately 50,000 scored bytes
from this pool using a fixed seed. `<corpus>_<MB>MB.split.json` records source, split
version, byte count, and checksums. The entire training file is available in every arm;
the ET subset stays inside this pool. No additional per-file tail split is applied.

Legacy arrays used per-file tail splits and must be rebuilt; the trainer rejects them
without the new split metadata. Preparation automatically rebuilds arrays whose split
metadata is missing. A pending publication record lets preparation finish interrupted
array/metadata writes without downloading the corpus again. Use `--overwrite_foundation`
to explicitly rebuild an already completed corpus, or choose a new `--out_dir`.
The builder refuses to replace an existing shared validation
file with different bytes; changed upstream data require a new directory for all scales.
Foundation preparation fails if the source cannot supply the requested budget.


For the complete 100M foundation Figure-6 replication, manual preparation is not
required: `python src/reproduce_foundation.py --scale 100M --output results/foundation_100M`
automatically downloads all six pinned public corpora, builds shared held-out
splits, trains/resumes, and generates verified figures and CSVs. Pinned dataset
revisions and parquet shard selection are in `src/foundation_sources.py`; no
private cache or manually supplied data is needed. See the root README.
