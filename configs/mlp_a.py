"""MLP config A with three paired seeds and post-hoc mechanism archives."""
DATASETS = ['mnist', 'fashion', 'kmnist']
WIDTHS = [8, 16, 32, 128, 256, 512, 1024, 2048]
SEEDS = [0, 1, 2]
ORDERINGS = ['goodfirst', 'junkfirst']


def generate():
    """Use the original target labels and fixed random-label OOD tasks."""
    return [('vision', dict(arch='mlp', task=task, width=width, depth=2,
                           ord=order, seed=seed, phase1=30000, phase2=120000,
                           junk_fraction=.25, junk_volume=2000, junk_not_refed=True,
                           lr=3e-4, batch=256, config='A'))
            for width in reversed(WIDTHS) for task in DATASETS
            for seed in SEEDS for order in ORDERINGS]
