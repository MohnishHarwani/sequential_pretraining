"""Two-phase curriculum campaign -> feeds the core-L figure and the diff figure.

EDIT the constants below to change the experiment. `generate()` expands them into the full
run list (driver, hyperparameters). run.py injects `data_dir` and executes each run.
"""

# ---- curriculum structure (shared by every model) ----
JUNK_FRACTION  = 0.25     # fraction of phase-2 steps that use junk
JUNK_NOT_REFED = True     # independently resample B; shared source inputs may overlap with A

# ---- the LR/batch configs the paper sweeps (become the rows of the core-L figure) ----
# The same three configs are used for MLP accuracy and mechanism experiments.
CONFIGS = {'A': dict(lr=3e-4, batch=256),
           'B': dict(lr=1e-3, batch=64),
           'C': dict(lr=5e-4, batch=32)}

SEEDS     = [0, 1, 2]
ORDERINGS = ['goodfirst', 'junkfirst']

# ---- MLP image models. Scale = width. ----
IMAGE_PHASE1, IMAGE_PHASE2 = 30000, 120000
IMAGE_DEPTH        = 2
IMAGE_JUNK_VOLUME  = 2000
IMAGE = {
    'mlp': dict(datasets=['mnist', 'fashion', 'kmnist'],
                widths=[8, 16, 32, 128, 256, 512, 1024, 2048]),
}


def generate():
    runs = []
    for cname, hp in CONFIGS.items():
        for arch, spec in IMAGE.items():
            for ds in spec['datasets']:
                for W in spec['widths']:
                    for o in ORDERINGS:
                        for s in SEEDS:
                            runs.append(('vision', dict(
                                arch=arch, task=ds, width=W, depth=IMAGE_DEPTH, ord=o, seed=s,
                                phase1=IMAGE_PHASE1, phase2=IMAGE_PHASE2, junk_fraction=JUNK_FRACTION,
                                junk_volume=IMAGE_JUNK_VOLUME, junk_not_refed=JUNK_NOT_REFED,
                                lr=hp['lr'], batch=hp['batch'], config=cname)))
    return runs
