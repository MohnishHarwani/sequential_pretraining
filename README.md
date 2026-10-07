# Sequential Pretraining Favors Large Models

This is the code repository corresponding to the paper: Sequential Pretraining Favors Large Models.

There are two sets of experiments:

- **MLP:** MNIST, Fashion-MNIST and KMNIST, eight model widths,
  target-first/OOD-first training, and mechanistic activation overlap.
- **Foundation models + Exposure Therapy:** byte-level models at 100M, 500M
  and 1B, trained on FineWeb/TinyStories paired with code, math or German/Finnish
  Wikipedia. Each size includes both orderings and OOD-first with 30% Exposure
  Therapy, using seeds 0/1/2 (72 runs per size).

## Run

Use Python 3.10+ on a POSIX system. MLPs support CPU or CUDA. Foundation training
requires a CUDA GPU with bf16 support; 40 GB or more VRAM is recommended.

```bash
pip install -r requirements.txt
python src/reproduce_mlp.py
python src/reproduce_foundation.py --scales 100M
```

Both commands automatically download and prepare public data, train, save
checkpoints, and generate PNG/PDF figures and CSVs. Foundation sizes can be any
subset (`--scales 500M 1B`) or all three (`--scales all`). Use `--output DIR` to
change the result directory, `--data-dir DIR` to change the data cache, and
`--dry-run` to inspect the plan without running it.

## Outputs

Paths below are relative to the repository; custom outputs use the same layout.

| Contents | Location |
|---|---|
| Prepared data | `data/` (MLP), `data/foundation/` (foundation) |
| MLP primacy and representational-overlap PNG/PDFs | `results/mlp/figures/` |
| MLP records; checkpoints and probe archives | `results/mlp/runs.jsonl`; `results/mlp/runs.jsonl.artifacts/` |
| MLP CSVs and report | `results/mlp/analysis/` |
| Foundation PNG/PDFs, one per size | `results/foundation/figures/` |
| Foundation checkpoints | `results/foundation/<size>/runs/` |
| Foundation records, CSVs and report | `results/foundation/<size>/analysis/` |

MLP curves show final validation-accuracy gaps. Overlap compares hidden-unit
activations on 512 OOD training images after Phase 1 with 512 held-out target
images after Phase 2, using layer-mean top-neuron Jaccard overlap in OOD-first runs.
Saved activation scores support new thresholds without retraining:

```bash
python src/reproduce_mlp.py --records results/mlp/runs.jsonl --fractions .01 .05 .10 .25 --output results/mlp_reanalysis
```

## Code

`configs/` contains experiment hyperparameters; `src/` contains preparation,
training, checkpointing and analysis; `figures/` contains plotting code.
