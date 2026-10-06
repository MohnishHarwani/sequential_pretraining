# Overparameterization and curriculum ordering

Code for MLP experiments, MLP mechanistic overlap, and foundation-model pretraining
on how model scale changes a network's
sensitivity to the *order* in which it sees clean vs. corrupt/out-of-distribution data.

A model is trained in **two phases** on a shared trunk:

- **Phase 1 (bias):** either the good task (*good-first*) or a fixed corrupt/OOD signal
  (*junk-first* / *OOD-first*).
- **Phase 2 (common):** the good/target task, with a fixed fraction of independently sampled junk/OOD mixed in
  deterministically. Both phases draw from the same source pool; overlap is allowed.
  Toy corruption sets use independently generated fixed random targets.
  In the foundation experiment both phases are the **same length** (equal-phase).

Model scale is controlled by width and depth. See the paper for interpretation of the results.

## The four figures

| figure | script | what it shows |
|--------|--------|---------------|
| **core-L** | `figures/fig_core_L.py` | MLP ordering gap: A/B/C rows, dataset columns, separate lines for seeds 0/1/2 |
| **diff** | `figures/fig_diff.py` | accuracy on the Phase-1 corruption set before and after target training |
| **mechanism** | `figures/fig_mechanism.py` | MLP training-to-training activation overlap and three-seed mean final accuracy gap |
| **foundation** | `figures/fig_foundation.py` | target and aggregate validation loss at 100M/500M/1B, with and without 30% Exposure Therapy |

## Layout

```
configs/     hyperparameters for each experiment  (edit these)
src/         trainers (train_vision for MLPs, train_foundation), run.py (local runner),
             prepare_data.py (builds MLP image and foundation data)
figures/     figure scripts and shared layouts; read run records or analysis CSVs
data/        documentation; downloaded/prepared data are generated locally and gitignored
results/     generated: <experiment>.jsonl, <experiment>.jsonl.artifacts/, figures/*.png
```

## Install

Use Python 3.10+ on Linux or another POSIX system (the runners use filesystem
locks). MLP training can use CPU or CUDA; foundation training requires a CUDA GPU
with bf16 support. Data preparation and figure generation run on CPU.

```bash
pip install -r requirements.txt        # torch, numpy, matplotlib, datasets
```

`datasets` is used by MLP data preparation, including automatic downloads invoked
by the training entry points. Foundation preparation downloads pinned Parquet
shards through `huggingface_hub` and reads them with `pyarrow`.
Reference dependency versions are listed at the top of
`requirements.txt`.

## MLP reproduction from a clean checkout

```bash
pip install -r requirements.txt
sh scripts/run_mlp_matched_overlap.sh
```

The runner downloads MNIST, Fashion-MNIST, and KMNIST from their public sources,
prepares the training data and additional validation images, and then runs A/B/C
with three seeds while saving phase checkpoints and overlap probes, then writes
the per-config overlap figures, A/B/C overlap grid, and A/B/C primacy seed grid
under `results/mlp_ABC/analysis/`. No dataset
files need to be supplied manually. Use `python src/run.py mlp_a` for config A
alone, followed by `python figures/fig_mechanism.py --records results/mlp_a.jsonl`.
Generated data and results are excluded from version control; the source
release contains the scripts and documentation.

## One-command foundation reproduction (Figure 6 at 100M)

After installing `requirements.txt`, run this single command on a CUDA GPU:

```bash
python src/reproduce_foundation.py --scale 100M --output results/foundation_100M
```

The script **downloads and prepares all six public corpora automatically**, then
trains all 72 runs (two targets × four OOD corpora × three arms × three seeds),
verifies results and checkpoints, and writes a Figure-6-style PNG/PDF plus per-seed
and summary CSVs in `results/foundation_100M/analysis/`. Data preparation and
analysis run on CPU; training requires a CUDA GPU with bf16 support.

The 100M model has 100,650,048 parameters, width 1088, seven layers and context
512. It uses 9,000 + 9,000 optimizer updates, Adam at 1.5e-4, warmup 1,000,
clip 0.5, effective batch 32, and a 75/25 target/OOD Phase-2 mixture. Arms are
target-first, OOD-first, and OOD-first with 30% Exposure Therapy using a fixed
10% target-training subset. The 10% subset remains available to ordinary training.

Data revisions and sorted shard order are pinned in `src/foundation_sources.py`.
Preparation writes checksums and a shared document-hash-held-out validation pool,
reused across arms, seeds, and scales. Raw byte budgets are 345 MB per target
(FineWeb/TinyStories) and 500 MB per OOD corpus (CodeSearchNet, OpenWebMath,
German/Finnish Wikipedia). Both the training bytes and validation metadata are
verified before reuse. Network interruption can be retried with the same command;
completed corpora reuse their verified arrays and downloaded shards remain cached.
Interrupted array/metadata publication is completed automatically on retry. Arrays
without split metadata are rebuilt from the pinned sources, never reused as training
data without verification. Interrupted campaign initialization can also be retried;
any old partial `code/` snapshot is preserved as `code.incomplete.<id>`.

Training defaults to `--microbatch 8` with gradient accumulation: the scientific
optimizer batch remains 32. Use a smaller microbatch if GPU memory requires it.
Final weights and periodic optimizer/RNG checkpoints are saved. Re-running the
**same command** skips completed runs and resumes interrupted ones. A changed
configuration requires a separate output directory. SIGUSR1 or SIGTERM requests
a checkpoint at the next optimizer step and exit with status 75; rerun the command
to resume.

A short real-data, actual-size test is available with `--smoke` and a separate
output directory. It executes all three arms for 40 steps, not publication runs.
For clusters, `--stage prepare`, `--stage init`, `--stage worker` and
`--stage analyze` expose the same pipeline. Initialization freezes source and
configuration under the campaign's `code/` directory. Run one or more copies of
`python CAMPAIGN/code/src/reproduce_foundation.py --stage worker --output CAMPAIGN`
to share a lock-protected work queue, then `--stage analyze` after all workers finish.
Calling `python src/reproduce_foundation.py --stage worker --output CAMPAIGN`
also verifies the saved source hashes and launches the saved worker in a fresh
Python process, so later edits to the repository do not change campaign training.
Data preparation, training, resume logic, verification and figure generation are
included in this repository.

## Reproduce

1. **Optionally prepare data separately** (MLP runners and the one-command
   foundation pipeline already do this automatically):

   ```bash
   python src/prepare_data.py --group vision
   python src/prepare_data.py --group foundation --scale 100M  # or 500M / 1B / all
   ```

   Formats and rebuilding older arrays are described in `data/README.md`.
2. **Run experiments** (writes `results/<experiment>.jsonl`):

   ```bash
   python src/run.py curriculum  --data_dir data     # -> core-L and diff figures
   python src/run.py mechanism   --data_dir data     # -> mechanism figure (+ activation probes)
   python src/run.py foundation  --data_dir data     # all foundation scales; prepare their data first
   ```

   Useful flags: `--dry_run` prints the run list without training; `--limit N` runs the first
   `N` only, at full training length. A GPU is used automatically when available.
   Re-running the same command preserves results and skips completed runs. `--limit`
   may be increased or removed later. Each complete result is published atomically;
   an interrupted toy run restarts individually, while foundation runs resume their
   optimizer/RNG checkpoints. A configuration manifest beside the JSONL prevents
   mixing changed hyperparameters or data paths: use a new `--out` for a new experiment.
   Older JSONL files without that manifest are left untouched; use a new output file
   to adopt this runner. Keep the manifest with its results. Concurrent invocations
   targeting the same output are serialized. Use the one-command foundation pipeline
   above when you also need frozen source code and automatic data preparation.

3. **Make the figures** (writes `results/figures/*.png`):

   ```bash
   python figures/fig_core_L.py
   python figures/fig_diff.py
   python figures/fig_mechanism.py
   python figures/fig_foundation.py
   ```

## Changing hyperparameters

Every hyperparameter is a named constant at the top of the matching `configs/*.py`; `generate()`
expands them into the run list. Experiments that were swept over multiple LR/batch settings
expose those as a dict you edit in one place, e.g. in `configs/curriculum.py`:

```python
CONFIGS = {'A': dict(lr=3e-4, batch=256),
           'B': dict(lr=1e-3, batch=64),
           'C': dict(lr=5e-4, batch=32)}
```

Add/remove a key to add/drop a config; change widths, datasets, seeds, phase lengths, junk
fraction the same way. Trainers can also be run one-off from the CLI, e.g.
`python src/train_vision.py --arch mlp --task mnist --width 256 --ord junkfirst`.

## Experimental details

Shared curriculum: two-phase training; junk = real in-distribution inputs with fixed random
targets (memorizable). Phase-1 set A and Phase-2 set B are independently sampled
from the same training-input pool, without enforcing disjointness. Each toy set has its own
fixed random targets; the original set A is not deliberately replayed. The legacy option
`junk_not_refed=True` means resample B, not exclude A's input identities. The **MLP** models put two readout heads on a shared
trunk (one for the target task, one for the junk targets). Neither head is swapped
or reset at the phase boundary; the optimizer is retained. Target-first enters
Phase 2 with a trained target head, while OOD-first enters it with an untrained
target head. The resulting accuracy gap measures whole-model ordering sensitivity,
not a controlled estimate of trunk-only effects. The **foundation** models use a
**single** head: target, OOD and Exposure Therapy batches all go through the same next-byte head. Phase-2 junk
fraction `CFRAC = 0.25`. Optimizer Adam. Metric = final full-held-out target-task accuracy (`val_acc_full`);
`L` uses this final value in both orderings, not a training-curve average.

**MLP models** (`train_vision.py`) — `depth` hidden ReLU layers of width `W`,
with two linear heads on the shared trunk.

| | value |
|---|---|
| phase 1 / phase 2 steps | 30000 / 120000 |
| depth | 2 |
| junk volume | 2000 |
| target labels | original dataset labels |
| widths | 8–2048 |
| datasets | MNIST / Fashion-MNIST / KMNIST |
| LR/batch configs | A 3e-4/256 · B 1e-3/64 · C 5e-4/32 |
| seeds | 0, 1, 2 |

MLPs use standard mean cross-entropy and Adam with betas (0.9, 0.999), epsilon
1e-8, no weight decay, and a constant learning rate. In Phase 2, every fourth
optimizer update uses an entire corruption batch; the other three use target
batches. This gives 30,000 corruption and 90,000 target updates. It is not a
within-batch mixture. A/B/C have equal update budgets, but their batch sizes mean
38.4M / 9.6M / 4.8M sample presentations across both phases and tasks; their epoch
and sample budgets are not equal.

*Train/validation split.* For each dataset, keep the original 56,000 training
images and 4,000 held-out images from its official 60,000-image training split.
Add 2,222 images from the official test split to validation, giving **56,000 train
/ 6,222 validation** (approximately 90/10; rounded to whole images). Both selections
use fixed seed 12345 and are shared across widths, configurations, seeds, and
orderings. No training image is moved to validation. Normalization uses training
data only. The remaining official test images are unused.
`python src/prepare_data.py --group vision` automatically prepares the additional
validation images, including when the original arrays already exist. Use a new
results output for this protocol; earlier results used only 4,000 validation images.

**Mechanism** (`configs/mechanism.py`) — the same A/B/C MLP curriculum, with
activation archives at the end of both phases. Default overlap figures use
OOD-first runs and the **same 512 corruption-training A inputs at both phases**
(`0J` versus `TJ`). Neurons are ranked by mean absolute post-ReLU activation,
and the top fraction's Jaccard overlap is averaged over hidden layers, then
seeds 0–2. For a layer of width W, select `max(1, round(W * fraction))` neurons.
Defaults are 1%, 5%, and 10%; arbitrary thresholds, including 25%, require no
retraining. Accuracy curves use the full 6,222-image validation set, independently
of the training probe used for overlap. Validation probe scores are also retained
for explicitly requested alternative analyses.

**Foundation models** (`train_foundation.py`) — byte-level Transformer, context 512, pre-LN,
heads `max(4, W//128)`, FFN `4W`, Adam (betas 0.9/0.999, no weight decay), LR 1.5e-4, 1000-step
warmup then constant, grad-clip 0.5, bf16. OOD = real OOD corpus text (real next-byte targets),
predicted through the **same single head** as the target -- there is no separate junk head and no
head switching at the phase boundary.

*Equal phases.* Phase 2 is the same length as phase 1, so a run stops at 2x the phase-1 budget:

| scale | width | layers | phase 1 = phase 2 | total steps | phase-2 OOD set |
|---|---|---|---|---|---|
| 100M | 1088 | 7 | 9,000 | 18,000 | 2,000 sequences |
| 500M | 2432 | 7 | 27,000 | 54,000 | 6,000 sequences |
| 1B | 3456 | 7 | 43,000 | 86,000 | 6,000 sequences |

*Phase 2 mixture.* Deterministic 75/25: every 4th optimization step is a full OOD batch
(`JUNK_FRACTION = 0.25`), the other three are target batches. The phase-2 OOD set is a fixed
draw from the same OOD training pool as Phase 1; overlap is allowed.

*Exposure Therapy (ET).* With probability `exposure_rate` a step is replaced by a batch
from a fixed slice of the target training pool. The slice **stays in the pool** and is also drawn
by ordinary target steps, so ET adds no data, holds nothing out, and adds no optimizer steps. It
applies to OOD-first runs only, in both phases. The exposure set contains `exposure_set_fraction` (default 10%) of the
complete target training sequences (rounded to whole sequences). Selecting it does not
shrink the ordinary training pool or change validation; all arms use the same full pool.
The standalone CLI exposes `--exposure_rate` and `--exposure_set_fraction`; campaign
defaults are `EXPOSURE_RATES` and `EXPOSURE_SET_FRACTION` in `configs/foundation.py`.
Results and checkpoints use these field names plus `exposure_in_training_pool=True`.

*Validation.* Approximately 50,000 scored bytes per distribution (100,000 total), as
**random** 512-byte sequences drawn without replacement from a shared per-corpus validation
file. A deterministic content-hash split reserves approximately 5% of source documents
before filling any training budget; those documents are excluded from training at every
scale. A fixed 1 MB prefix of each held-out stream is saved, and a fixed sampling seed
selects the same evaluation sequences for every run, arm, and scale using that corpus.
Whole sequences are retained; the small rounding above 50,000 is intentional. Reported per run:
`val_ce` (target), `ood_val_ce` (OOD), and `aggregate` = (target NLL + OOD NLL) / total scored
bytes, i.e. the two distributions weighted equally.

Arms per (target corpus, OOD corpus, seed): target-first, OOD-first, OOD-first + Exposure Therapy
(30%). Target corpora: fineweb, tinystories. OOD corpora: code, math, dewiki, fiwiki. Phase-1 OOD
uses half the single-pass budget, cycled (`P1_FRACTION = 0.5`). Seeds 0–2.

*Checkpoints.* See the Checkpoints section below; foundation tags are
`<scale>_<target>_<ood>_<ord>_et<exposure_rate>_s<seed>_W<width>`.

## MLP config A and later overlap analysis

`configs/mlp_a.py` contains the config A campaign: MNIST, Fashion-MNIST,
KMNIST; widths 8, 16, 32, 128, 256, 512, 1024, 2048; both orderings; seeds 0–2.
It retains LR 3e-4, batch 256, depth 2, 30,000/120,000 phase lengths, original target
labels, and 25% Phase-2 corruption. This is 144 runs.

Whenever an MLP run saves checkpoints, it also saves Phase-1 weights
(`.phase1.pt`), exact standardized probe inputs, source indices, normalization
statistics, corruption targets (`.mechanism.npz`), and both-phase activation
snapshots. This applies to both orderings, including ordinary curriculum runs.
Each result's `mechanism` field lists the artifact paths. These support overlap
analysis later, including different top-unit thresholds, without retraining.

For a completed-run JSON record:

```bash
python src/compute_overlap.py path/to/run.json
python src/compute_overlap.py path/to/run.json --recompute /tmp/recomputed.npz
python src/compute_overlap.py path/to/run.json --fractions .01 .05 .10 .25
python src/compute_overlap.py path/to/run.json --comparison validation --fractions .25
```

The second command reconstructs activations from both phase checkpoints and the
saved inputs. It defaults to CPU; `--device cuda` uses an available CUDA GPU.
The metric is mean layer-wise Jaccard overlap of top 1%, 5%, and 10% units ranked
by mean absolute post-ReLU activation on the same corruption-training probe at both phases.

The default `training` comparison uses the same saved corruption-set-A inputs
at both phases. Explicit alternatives are `validation` (same held-out inputs at
both phases) and `cross_probe` (Phase-1 corruption inputs versus final validation
inputs). These alternatives are labeled separately and are not used in the
default mechanism figures. All neuron scores are saved, so thresholds can be changed
without retraining. Each default probe contains 512 images; these are not full-corpus
overlaps. Earlier campaigns retain their original probes after a data-split change.

For completed campaign directories with manifests and per-run records, generate
matched-probe plots and per-seed/per-layer CSVs in a new directory:

```bash
python src/analyze_probe_overlap.py CAMPAIGN [CAMPAIGN ...] --output results/matched_probes
```

This config A analysis includes both orderings, plots matched and original overlaps,
and spot-checks saved activations against CPU inference from phase checkpoints.

## Checkpoints

**Every run in every experiment keeps its final weights.** At the end of training each trainer
writes `<output.jsonl>.artifacts/<run-id>/<tag>.pt` when using `src/run.py`, containing:

| key | contents |
|---|---|
| `model` | model `state_dict` |
| `step` | total steps trained |
| `meta` | the full run configuration (scale/arch, task or corpora, arm, exposure rate and set fraction, phases, LR, batch, seeds) |

The write goes through a `.tmp` file and an atomic rename, so a killed job cannot leave a corrupt
file. MLPs additionally keep Phase-1 weights and mechanism inputs/snapshots as
described above. Foundation runs also save periodic optimizer/RNG state for exact continuation; the other portable trainers retain final weights as described above.

Keep these files. A loss value recorded during training can never be re-scored on a different
validation set, a different metric, or with activation probes — only the weights can, and at
foundation scale re-running a single arm is many GPU-hours. Tags:

```
foundation   <scale>_<target>_<ood>_<ord>_et<exposure_rate>_s<seed>_W<width>
vision       <config>_<arch>_<task>_W<width>_<ord>_s<seed>
```

`--no_ckpt` disables saving; `--ckpt_dir DIR` writes under
`DIR/<output-id>/<run-id>/`. Separate output files have separate artifact directories,
even when model tags coincide. Each result record carries its own `checkpoint` path.
The one-command foundation pipeline instead keeps artifacts under `CAMPAIGN/runs/`.

## Notes

- The paper's classification ET dose sweep (Figure 5 / Appendix F) is illustrative
  hyperparameter-tuning guidance and is intentionally outside this code release.

- Results are written incrementally to `results/<experiment>.jsonl` (one JSON record per run);
  the figure scripts read those. Point elsewhere with the `REPRO_RESULTS` env var.
- The foundation experiment is genuine billion-parameter pretraining and is included for
  completeness of the record, not for quick reruns.

MLP target-task training uses the original dataset labels. The separate junk/OOD
sets retain their fixed random labels and their own output head.

To reproduce the matched-probe overlap comparison for **all A/B/C hyperparameter
sets and seeds 0/1/2**, including automatic image downloads, training,
saved phase/final checkpoints, and figures:

```bash
sh scripts/run_mlp_matched_overlap.sh
```

This uses 56,000 training images and 6,222 validation images per dataset. Each
configuration gets `A/mlp_training_overlap.png` (or B/C), with threshold columns
and three-seed mean curves for all datasets. It also writes
`mlp_ABC_training_overlap.png` with config rows and threshold columns, and
`mlp_primacy_seeds.png` with config rows, dataset columns, and separate seed lines.
Every figure has a matching PDF. Overlap plots use training-to-training probes;
the CSVs retain explicitly labeled alternative comparisons. Increasing the
accuracy validation set does not enlarge the fixed 512-image probes.

Generate these outputs, including top 25%, from existing run records:

```bash
python src/analyze_mlp_configs.py RESULTS.jsonl --output PLOT_DIR --fractions .01 .05 .10 .25
```

Or export specific layouts directly from saved records:

```bash
# Three dataset columns, one config A row, separate seed lines (Figure 2 layout).
python figures/fig_core_L.py --records RESULTS.jsonl --config A --output results/figures/primacy_A
# Three configuration rows and three dataset columns, separate seed lines.
python figures/fig_core_L.py --records RESULTS.jsonl --output results/figures/primacy_ABC
# Config A, one top-25% overlap panel (Figure 3 layout).
python figures/fig_mechanism.py --records RESULTS.jsonl --config A --fractions .25 --output results/figures/overlap_A_top25
# Config rows and overlap-threshold columns (Figure 7 layout with overlap).
python figures/fig_mechanism.py --records RESULTS.jsonl --config all --output results/figures/overlap_ABC
```

These commands use saved activation scores and final accuracies; they do not
train models or alter the input results.
