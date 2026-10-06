"""Foundation-model ordering campaign (100M / 500M / 1B) -> feeds the foundation figure.

WARNING: this is A100 / billion-parameter-scale compute (each 1B run is many GPU-hours over
billions of bytes). The code is here for full reproducibility, not quick iteration.

Each (target corpus, OOD corpus, scale, seed) is run in three arms:
  target-first, OOD-first, OOD-first + Exposure Therapy (30% exposure rate).
Phases are EQUAL (phase2 == phase1); phase 2 intermixes OOD deterministically at 1 step in 4.
EDIT the constants below to change the experiment.
"""
SEQ    = 512
LAYERS = 7
JUNK_FRACTION = 0.25          # phase 2: every 4th step is an OOD batch (deterministic 75/25)
P1_FRACTION   = 0.5           # phase-1 OOD uses half the single-pass budget, cycled (memorization)

# Exposure set: 10% of the target TRAINING sequences. It remains in the training
# pool; every arm uses the same full training pool, with no ET-dependent holdout.
EXPOSURE_SET_FRACTION = 0.10

# optimizer (shared across scales): Adam, betas (0.9, 0.999), no weight decay,
# linear warmup then constant LR
LR, WARMUP, CLIP, BATCH = 1.5e-4, 1000, 0.5, 32

SEEDS        = [0, 1, 2]
EXPOSURE_RATES = [0.30]         # Exposure Therapy dose(s); add 0.15 to sweep

# scale -> width, equal phase length, phase-2 OOD set size, and the corpus byte budgets in MB.
# Corpus files are named <corpus>_<MB>MB.npy and are built by src/prepare_data.py.
SCALES = {
    '100M': dict(width=1088, phase=9000,  junk_volume=2000, target_mb=345,  ood_mb=500),
    '500M': dict(width=2432, phase=27000, junk_volume=6000, target_mb=1035, ood_mb=1500),
    '1B':   dict(width=3456, phase=43000, junk_volume=6000, target_mb=1665, ood_mb=1500),
}
GOOD_CORPORA = ['fineweb', 'tinystories']
OOD_CORPORA  = ['code', 'math', 'dewiki', 'fiwiki']


def generate(scale=None):
    if scale is not None and scale not in SCALES:
        raise ValueError(f"Unknown foundation scale: {scale}")
    runs = []
    for sc, spec in SCALES.items():
        if scale is not None and sc != scale:
            continue
        for good in GOOD_CORPORA:
            for ood in OOD_CORPORA:
                for s in SEEDS:
                    arms = [('goodfirst', 0.0), ('junkfirst', 0.0)] + [('junkfirst', rate) for rate in EXPOSURE_RATES]
                    for ordr, exposure_rate in arms:
                        runs.append(('foundation', dict(
                            width=spec['width'], layers=LAYERS, seq=SEQ, ord=ordr, seed=s,
                            phase1=spec['phase'], phase2=spec['phase'], junk_fraction=JUNK_FRACTION,
                            junk_volume=spec['junk_volume'], exposure_rate=exposure_rate,
                            exposure_set_fraction=EXPOSURE_SET_FRACTION,
                            p1_fraction=P1_FRACTION, lr=LR, warmup=WARMUP, clip=CLIP, batch=BATCH,
                            good_data=f"{good}_{spec['target_mb']}MB.npy", ood_data=f"{ood}_{spec['ood_mb']}MB.npy",
                            config=sc)))
    return runs
