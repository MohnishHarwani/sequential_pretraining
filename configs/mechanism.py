"""Mechanism campaign -> feeds the mechanism figure.

Same two-phase curriculum, but with activation probes recorded at the end of phase 1 and the
end of phase 2. The overlap metric compares the top neurons on the same fixed
corruption-training probe at both phases, using layer-mean Jaccard overlap.

Both orderings are run (good-first supplies the L baseline); probes are captured for
both when checkpoints are enabled. The mechanism figure uses junk-first overlap.
EDIT the constants below to change the experiment.
"""
DATASETS = ['mnist', 'fashion', 'kmnist']
WIDTHS   = [8, 16, 32, 128, 256, 512, 1024, 2048]
SEEDS    = [0, 1, 2]

PHASE1, PHASE2 = 30000, 120000
DEPTH          = 2
JUNK_FRACTION  = 0.25
JUNK_VOLUME    = 2000

# The same three LR/batch configs as the curriculum campaign.
CONFIGS = {'A': dict(lr=3e-4, batch=256),
           'B': dict(lr=1e-3, batch=64),
           'C': dict(lr=5e-4, batch=32)}

TOP_FRACTIONS = (0.01, 0.05, 0.10)  # CLI --fractions overrides these without retraining


def generate():
    runs = []
    for cname, hp in CONFIGS.items():
        for ds in DATASETS:
            for W in WIDTHS:
                for s in SEEDS:
                    for o in ['junkfirst', 'goodfirst']:
                        runs.append(('vision', dict(
                            arch='mlp', task=ds, width=W, depth=DEPTH, ord=o, seed=s,
                            phase1=PHASE1, phase2=PHASE2, junk_fraction=JUNK_FRACTION,
                            junk_volume=JUNK_VOLUME, junk_not_refed=True,
                            lr=hp['lr'], batch=hp['batch'], config=cname)))
    return runs
