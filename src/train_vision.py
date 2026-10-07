"""Two-phase MLP trainer for primacy, corruption-retention, and mechanistic overlap figures.

Experiment: a shared trunk with two linear heads (a "good" head and a "junk" head).
Phase 1 ("bias") trains on either the good task (good-first) or a fixed set of memorizable
junk examples (junk-first). Phase 2 ("common") trains on the good task with independently sampled junk set B.
Both sets draw from the same input pool, allowing overlap, with independent fixed random
targets. JUNK_NOT_REFED means resample B instead of deliberately replaying set A.

Scale is set by width `W` (and depth). The experiment measures sensitivity to
phase-1 ordering; monotonicity with model size is not assumed.

Public API:
    run(cfg: dict) -> dict          # trains one model, returns a JSON-able result record
MLP runs automatically retain Phase-1 and final weights, exact probe inputs,
and both-phase activation snapshots. cfg['probe_out'] can override the snapshot directory.
"""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from image_data import flatten_images, mlp_split_indices, load_extra_validation, MLP_SPLIT_VERSION
from mechanism_archive import MechanismArchive

# ------------------------------------------------------------------ data
def load_dataset(task, data_dir):
    """Expects <data_dir>/<task>_x.npy (float, [N, ...]) and <task>_y.npy (int, [N]).

    Returns RAW features: standardization happens after the train/validation split, using
    training statistics only (see run()).
    """
    x = np.load(os.path.join(data_dir, f"{task}_x.npy"))
    y = np.load(os.path.join(data_dir, f"{task}_y.npy")).astype(np.int64)
    return flatten_images(x, task), y

# ------------------------------------------------------------------ models
class MLP(nn.Module):
    """`depth` hidden ReLU layers of width W; shared trunk, separate good/junk heads."""
    def __init__(self, din, dout, W, depth):
        super().__init__()
        self.inp = nn.Linear(din, W)
        self.hid = nn.ModuleList([nn.Linear(W, W) for _ in range(max(0, depth - 1))])
        self.hg, self.hj = nn.Linear(W, dout), nn.Linear(W, dout)
    def trunk_acts(self, x):                              # returns post-ReLU activation of every hidden layer
        acts = []
        h = F.relu(self.inp(x)); acts.append(h)
        for l in self.hid:
            h = F.relu(l(h)); acts.append(h)
        return h, acts
    def trunk(self, x): return self.trunk_acts(x)[0]
    def good(self, x): return self.hg(self.trunk(x))
    def junk(self, x): return self.hj(self.trunk(x))

# ------------------------------------------------------------------ training
def run(cfg):
    """cfg keys: arch task width depth ord seed phase1 phase2 junk_fraction junk_volume
    junk_not_refed lr batch config data_dir [probe_out] [log_every] [ckpt_dir]."""
    if cfg.get('ord') not in ('goodfirst', 'junkfirst'):
        raise ValueError(f"ord must be 'goodfirst' or 'junkfirst', got {cfg.get('ord')!r}")
    from training_setup import prepare_training
    cfg = prepare_training(cfg, 'vision')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch, task, W, depth = cfg['arch'], cfg['task'], cfg['width'], cfg['depth']
    if arch != 'mlp':
        raise ValueError('This trainer supports only MLPs')
    if cfg.get('validation_protocol', MLP_SPLIT_VERSION) != MLP_SPLIT_VERSION:
        raise ValueError('Unsupported MLP validation protocol')
    ORD, SEED = cfg['ord'], cfg['seed']
    BIAS, COMMON = cfg['phase1'], cfg['phase2']
    CFRAC, NJUNK = cfg['junk_fraction'], cfg['junk_volume']
    LR, B = cfg['lr'], cfg['batch']
    LOG = cfg.get('log_every', 250)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)

    x, y = load_dataset(task, cfg['data_dir'])
    din, dout = x.shape[1], int(y.max() + 1)
    # Preserve all 56,000 training images and the original 4,000 validation images.
    # Add 2,222 official test images to validation only: approximately 90/10.
    tr_idx, original_val_idx = mlp_split_indices(len(x))
    extra_x, extra_y, extra_test_idx, extra_meta = load_extra_validation(task, cfg['data_dir'], len(x))
    val_idx = np.concatenate([original_val_idx, len(x) + np.arange(len(extra_x))])
    Xtr, Ytr = x[tr_idx], y[tr_idx].copy()
    Vx = np.concatenate([x[original_val_idx], extra_x])
    Vy = np.concatenate([y[original_val_idx], extra_y])
    ntr = len(Xtr)
    validation = dict(protocol=MLP_SPLIT_VERSION, training_count=ntr, validation_count=len(Vx),
                      original_validation_count=len(original_val_idx), added_validation_count=len(extra_x),
                      extra_source=extra_meta)
    # Standardize with TRAINING statistics only; the held-out set is transformed with them but
    # never contributes to them.
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Vx = (Vx - mu) / sd
    def good_batch(bs):
        ii = rng.integers(0, ntr, bs); return Xtr[ii], Ytr[ii]

    # junk set A (phase 1): real in-distribution inputs with fixed random labels
    junk_a_indices = rng.integers(0, ntr, NJUNK)
    JxA = np.asarray(Xtr[junk_a_indices], np.float32).copy()
    JyA = rng.integers(0, dout, NJUNK).astype(np.int64)
    if cfg['junk_not_refed']:                             # independent draw; input overlap allowed
        r2 = np.random.default_rng(SEED + 777)
        JxB = np.asarray(good_batch(NJUNK)[0], np.float32).copy()
        JyB = r2.integers(0, dout, NJUNK).astype(np.int64)
    else:
        JxB, JyB = JxA, JyA

    net = MLP(din, dout, W, depth).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    params_M = round(sum(p.numel() for p in net.parameters()) / 1e6, 4)

    Vx_t, Vy_t = torch.tensor(Vx, device=dev), torch.tensor(Vy, device=dev)
    JA_x, JA_y = torch.tensor(JxA, device=dev), torch.tensor(JyA, device=dev)

    def src(step):
        if step < BIAS:
            return 'g' if ORD == 'goodfirst' else 'j'
        K = max(2, round(1.0 / CFRAC))                    # phase-2 junk fraction = CFRAC
        return 'j' if step % K == 0 else 'g'

    @torch.no_grad()
    def _acc(head, X, Y):
        """Accuracy in evaluation mode, preserving the previous training mode."""
        was_training = net.training
        net.eval()
        try:
            return (head(X).argmax(-1) == Y).float().mean().item()
        finally:
            net.train(was_training)

    def junkA_acc():
        return _acc(net.junk, JA_x, JA_y)

    def val_acc():
        return _acc(net.good, Vx_t, Vy_t)

    # MLP checkpoints retain both phases and exact probe inputs for later analysis.
    CKDIR = cfg.get('ckpt_dir')
    tag = '%s_%s_%s_W%d_%s_s%d' % (cfg['config'], arch, task, W, ORD, SEED)
    ckpt = os.path.join(CKDIR, tag + '.pt') if CKDIR else None
    probe_out = cfg.get('probe_out') or (os.path.join(CKDIR, 'mechanism_probes') if CKDIR else None)
    do_probe = probe_out is not None
    archive = None
    snap = {}
    if do_probe:
        PN = min(512, NJUNK, len(Vx))
        Jp = torch.tensor(JxA[:PN], device=dev)
        Xp = torch.tensor(Vx[:PN], device=dev)
        archive = MechanismArchive(cfg, tag, ckpt, probe_out, dict(
            junk_probe=JxA[:PN], target_probe=Vx[:PN],
            junk_probe_targets=JyA[:PN], target_probe_targets=Vy[:PN],
            junk_source_indices=tr_idx[junk_a_indices[:PN]], target_source_indices=val_idx[:PN],
            validation_indices=val_idx, training_indices=tr_idx,
            original_validation_indices=original_val_idx, additional_validation_test_indices=extra_test_idx,
            normalization_mean=mu, normalization_std=sd,
            phase1_junk_indices=tr_idx[junk_a_indices], phase1_junk_targets=JyA))
        @torch.no_grad()
        def capture(phase):
            was_training = net.training
            net.eval()
            name, probe = ('0J', Jp) if phase == '0' else ('TX', Xp)
            _, acts = net.trunk_acts(probe)
            snap[name] = [a.abs().mean(0).cpu().numpy() for a in acts]
            net.train(was_training)
            archive.save_activations(snap)

    def save():
        if not ckpt: return
        os.makedirs(CKDIR, exist_ok=True)
        torch.save(dict(model=net.state_dict(), step=BIAS + COMMON,
                        meta=dict(config=cfg['config'], arch=arch, task=task, width=W, depth=depth,
                                  ord=ORD, seed=SEED, phase1=BIAS, phase2=COMMON,
                                  junk_fraction=CFRAC, junk_volume=NJUNK,
                                  lr=LR, batch=B, junk_not_refed=cfg['junk_not_refed'], validation=validation,
                                  **({'mechanism': archive.paths()} if archive else {}))), ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)

    steps, vacc, jcurve = [], [], []
    junk_bias = None
    for step in range(BIAS + COMMON):
        s = src(step)
        if s == 'g':
            bx, by = good_batch(B); out = net.good(torch.tensor(np.asarray(bx, np.float32), device=dev))
            yt = torch.tensor(by, device=dev)
        else:
            Jx, Jy = (JxA, JyA) if step < BIAS else (JxB, JyB)
            ii = rng.integers(0, NJUNK, B)
            out = net.junk(torch.tensor(Jx[ii], device=dev)); yt = torch.tensor(Jy[ii], device=dev)
        loss = F.cross_entropy(out, yt)
        opt.zero_grad(); loss.backward(); opt.step()

        if step == BIAS - 1:
            junk_bias = round(junkA_acc(), 4)             # memorization at the phase boundary
            if do_probe:
                capture('0')
                archive.save_phase1(net, BIAS)
        if step % LOG == 0 or step == BIAS + COMMON - 1:
            steps.append(step); vacc.append(round(val_acc(), 4)); jcurve.append(round(junkA_acc(), 4))
    junk_end = round(junkA_acc(), 4)
    if do_probe:
        capture('T')
    save()

    return dict(arch=arch, task=task, width=W, depth=depth, params_M=params_M,
                ord=ORD, seed=SEED, config=cfg['config'], lr=LR, batch=B,
                phase1=BIAS, phase2=COMMON, junk_fraction=CFRAC, junk_volume=NJUNK,
                steps=steps, vacc=vacc, jcurve=jcurve,
                val_acc_full=round(val_acc(), 4), validation=validation,
                junk_bias=junk_bias, junk_end=junk_end, checkpoint=ckpt,
                **({'mechanism': archive.paths()} if archive else {}))


if __name__ == '__main__':                                # optional standalone use
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['mlp'], default='mlp'); ap.add_argument('--task', choices=['mnist', 'fashion', 'kmnist'], default='mnist')
    ap.add_argument('--width', type=int, default=8); ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--ord', choices=['goodfirst', 'junkfirst'], default='junkfirst'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--phase1', type=int, default=30000); ap.add_argument('--phase2', type=int, default=120000)
    ap.add_argument('--junk_fraction', type=float, default=0.25); ap.add_argument('--junk_volume', type=int, default=2000)
    ap.add_argument('--junk_not_refed', type=int, default=1)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--batch', type=int, default=256)
    ap.set_defaults(config='mlp')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--probe_out', default=None)
    ap.add_argument('--ckpt_dir', default=None, help='override the automatic per-configuration checkpoint directory')
    a = vars(ap.parse_args()); a['junk_not_refed'] = bool(a['junk_not_refed'])
    print(json.dumps(run(a)))
