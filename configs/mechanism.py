"""Mechanism campaign -> feeds the mechanism figure.

Same two-phase curriculum, but with activation probes recorded at the end of phase 1 and the
end of phase 2. The overlap metric compares the top neurons on an OOD probe
after phase 1 with a held-out target probe after phase 2, using layer-mean Jaccard overlap.

Both orderings are run (good-first supplies the baseline); probes are captured for
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

# The mechanism experiment uses the same learning rate and batch size as the curriculum campaign.
LEARNING_RATE = 3e-4
BATCH_SIZE = 256

TOP_FRACTIONS = (0.01, 0.05, 0.10)  # CLI --fractions overrides these without retraining


def generate():
    runs = []
    for ds in DATASETS:
        for W in WIDTHS:
            for s in SEEDS:
                for o in ['junkfirst', 'goodfirst']:
                    runs.append(('vision', dict(
                        arch='mlp', task=ds, width=W, depth=DEPTH, ord=o, seed=s,
                        phase1=PHASE1, phase2=PHASE2, junk_fraction=JUNK_FRACTION,
                        junk_volume=JUNK_VOLUME, junk_not_refed=True,
                        lr=LEARNING_RATE, batch=BATCH_SIZE, config='mlp')))
    return runs
