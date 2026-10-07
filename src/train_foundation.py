"""Byte-level pretraining driver for the foundation-model (100M / 500M / 1B) ordering experiment.

A byte-level Transformer LM is pretrained in two EQUAL phases:
  phase 1 ("bias"):   either the target corpus (target-first) or an OUT-OF-DISTRIBUTION corpus
                      (OOD-first) -- code / math / German / Finnish text.
  phase 2 ("common"): the target corpus with a deterministic 75/25 target/OOD intermix --
                      every 4th optimization step is a full OOD batch (junk_fraction = 0.25).
Exposure Therapy (ET) replaces a fraction EXPOSURE_RATE of the steps with a batch from a fixed
SAME-SET slice of the target training pool: the slice stays in the ordinary training pool, so no
data is added or held out. It is applied in OOD-first runs only, in both phases.

Reported quantities (nats per scored byte):
  val_ce      target-corpus validation loss
  ood_val_ce  OOD-corpus validation loss
  aggregate   (target NLL + OOD NLL) / total scored bytes  -- equal bytes per distribution

Validation data: approximately VAL_BYTES scored bytes per distribution, drawn as random
512-byte sequences from a shared per-corpus held-out file. Document-level hash splitting
excludes these documents from training at every scale. The same sampling seed and shared
file give identical validation sequences across runs, arms, and model scales.

Checkpoints: FINAL weights are written to <ckpt_dir>/<tag>.pt automatically, so a
finished run can be re-scored on a new validation set without retraining -- at this scale,
re-running is many GPU-hours.
Periodic .pt.resume checkpoints retain optimizer/RNG state for interrupted runs.
Public API: run(cfg: dict) -> dict.
"""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from foundation_data import exposure_count, load_corpus
from runtime import TrainingState, progress

V = 256
VAL_BYTES = 50_000          # per distribution; 100,000 bytes of validation data in total
VAL_SEED = 12345            # fixed: the same validation sequences for every run


class GPT(nn.Module):
    def __init__(self, W, nl, seq):
        super().__init__()
        self.tok = nn.Embedding(V, W); self.pos = nn.Embedding(seq, W)
        enc = nn.TransformerEncoderLayer(W, max(4, W // 128), W * 4, batch_first=True,
                                         dropout=0.0, activation='gelu', norm_first=True)
        self.tr = nn.TransformerEncoder(enc, nl); self.ln = nn.LayerNorm(W)
        self.head = nn.Linear(W, V)
        self.register_buffer('m', torch.triu(torch.ones(seq, seq) * float('-inf'), 1))
    def forward(self, x):
        p = torch.arange(x.shape[1], device=x.device)
        h = self.tok(x) + self.pos(p)[None]
        return self.head(self.ln(self.tr(h, mask=self.m[:x.shape[1], :x.shape[1]])))


def _docs(a, seq, dev):
    """Contiguous 512-byte sequences (the training pool)."""
    n = (len(a) // seq) * seq
    return torch.tensor(np.asarray(a[:n], np.uint8).reshape(-1, seq), dtype=torch.uint8, device=dev)


def _val_set(arr, start, seq, dev, n_bytes=VAL_BYTES, seed=VAL_SEED):
    """Random sequence-grid samples from the shared held-out array arr[start:].

    n_bytes / seq sequences are sampled without replacement from the held-out sequence grid with a
    fixed seed, so all runs are scored on identical held-out data.
    """
    n_seq = int(np.ceil(n_bytes / seq))
    grid = (len(arr) - start) // seq
    if grid < n_seq:
        raise ValueError(f'held-out pool has {grid} sequences, need {n_seq}')
    idx = np.random.default_rng(seed).choice(grid, size=n_seq, replace=False)
    rows = [np.asarray(arr[start + i * seq: start + (i + 1) * seq], np.uint8) for i in sorted(idx)]
    return torch.tensor(np.stack(rows), dtype=torch.uint8, device=dev)


def run(cfg):
    """cfg keys: width layers seq ord seed phase1 phase2 junk_fraction junk_volume
    exposure_rate exposure_set_fraction p1_fraction lr warmup clip batch good_data ood_data config data_dir
    [log_every] [ckpt_dir]."""
    if cfg.get('ord') not in ('goodfirst', 'junkfirst'):
        raise ValueError(f"ord must be 'goodfirst' or 'junkfirst', got {cfg.get('ord')!r}")
    from training_setup import prepare_training
    cfg = prepare_training(cfg, 'foundation')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda'))
    W, NL, SEQ = cfg['width'], cfg['layers'], cfg['seq']
    ORD, SEED = cfg['ord'], cfg['seed']
    BIAS, COMMON = cfg['phase1'], cfg['phase2']
    CFRAC, M = cfg['junk_fraction'], cfg['junk_volume']
    EXPOSURE_RATE, EXPOSURE_SET_FRACTION, P1FRAC = cfg['exposure_rate'], cfg['exposure_set_fraction'], cfg['p1_fraction']
    LR, WARM, CLIP, B = cfg['lr'], cfg['warmup'], cfg['clip'], cfg['batch']
    LOG = cfg.get('log_every', 500)
    MICRO = int(cfg.get('microbatch', B))
    if MICRO < 1 or MICRO > B:
        raise ValueError('microbatch must be between 1 and batch')
    assert BIAS == COMMON, f'phases must be equal: got {BIAS} and {COMMON}'
    K = round(1.0 / CFRAC)
    assert abs(1.0 / K - CFRAC) < 1e-9 and K >= 2, 'junk_fraction must be 1/K (0.25 -> every 4th step)'
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED); r2 = np.random.default_rng(SEED + 555)

    # Full training-only pool in every arm. Validation was separated by document
    # hash during preparation, before applying any scale-specific byte budget.
    good, good_val, good_meta = load_corpus(os.path.join(cfg['data_dir'], cfg['good_data']))
    g_tr = _docs(good, SEQ, dev)
    g_te = _val_set(good_val, 0, SEQ, dev)
    n_exposure = exposure_count(g_tr.shape[0], EXPOSURE_SET_FRACTION, EXPOSURE_RATE)

    # Exposure set: a slice of the TRAINED pool. Its sequences stay in the pool and are also drawn
    # by ordinary target steps, so Exposure Therapy adds no data and withholds none.
    EXPOSURE_SET = g_tr[-n_exposure:] if n_exposure else None

    ood, ood_val, ood_meta = load_corpus(os.path.join(cfg['data_dir'], cfg['ood_data']))
    d2 = _docs(ood, SEQ, dev)
    ood_te = _val_set(ood_val, 0, SEQ, dev)
    # Independently sampled from the same pool; overlap with Phase 1 is allowed.
    JxB = d2[torch.as_tensor(r2.integers(0, d2.shape[0], M), device=dev)].clone()

    # phase-1 OOD: a fixed permuted subset, walked in order and cycled
    # (p1_fraction < 1 => fewer unique sequences, each seen multiple times)
    pool = min(d2.shape[0], max(B, int(round(BIAS * B * P1FRAC))))
    p1 = np.random.default_rng(SEED + 999).permutation(d2.shape[0])[:pool]

    net = GPT(W, NL, SEQ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)      # Adam, betas (0.9, 0.999), no weight decay
    params_M = round(sum(p.numel() for p in net.parameters()) / 1e6, 2)

    # ---- final weights plus periodic optimizer/RNG checkpoints below ----
    CKDIR = cfg.get('ckpt_dir')
    tag = ('%s_%s_%s_%s_et%g_s%d_W%d' % (cfg['config'], os.path.splitext(cfg['good_data'])[0],
                                        os.path.splitext(cfg['ood_data'])[0], ORD, EXPOSURE_RATE, SEED, W))
    ckpt = os.path.join(CKDIR, tag + '.pt') if CKDIR else None

    def save():
        if not ckpt: return
        os.makedirs(CKDIR, exist_ok=True)
        torch.save(dict(model=net.state_dict(), step=BIAS + COMMON,
                        meta=dict(config=cfg['config'], width=W, layers=NL, seq=SEQ, ord=ORD, seed=SEED,
                                  exposure_rate=EXPOSURE_RATE, exposure_set_fraction=EXPOSURE_SET_FRACTION, exposure_in_training_pool=True,
                                  training_sequences=int(g_tr.shape[0]), exposure_sequences=n_exposure,
                                  validation=dict(target=good_meta, ood=ood_meta),
                                  phase1=BIAS, phase2=COMMON, junk_fraction=CFRAC, p1_fraction=P1FRAC,
                                  good_data=cfg['good_data'], ood_data=cfg['ood_data'],
                                  optimizer='adam', lr=LR, warmup=WARM, clip=CLIP, batch=B, microbatch=MICRO)),
                   ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)

    def lr_at(s):                                   # linear warmup then constant (no decay)
        return LR * (s + 1) / WARM if s < WARM else LR

    def _ce(x):
        with amp:
            return F.cross_entropy(net(x)[:, :-1].reshape(-1, V).float(), x[:, 1:].reshape(-1))

    def gstep():
        return g_tr[rng.integers(0, g_tr.shape[0], B)].long()

    def jstep(step):
        if step < BIAS:                             # phase 1: walk the cycled OOD subset
            idx = torch.as_tensor(p1[(step * B + np.arange(B)) % len(p1)].copy(), device=dev)
            return d2[idx].long()
        return JxB[rng.integers(0, JxB.shape[0], B)].long()   # phase 2: fixed OOD set

    def exposure_step():
        return EXPOSURE_SET[rng.integers(0, EXPOSURE_SET.shape[0], B)].long()

    def src(step):
        # Exposure Therapy replaces a step (it never adds one), in OOD-first runs, in both phases.
        if EXPOSURE_RATE > 0 and ORD == 'junkfirst' and rng.random() < EXPOSURE_RATE: return 'et'
        if step < BIAS: return 'g' if ORD == 'goodfirst' else 'j'
        return 'j' if step % K == 0 else 'g'        # phase 2: deterministic 75/25 target/OOD

    @torch.no_grad()
    def eval_ce(D):
        """Byte-weighted CE over the whole validation set (nats/byte) and its scored-byte count."""
        t = 0; lo = 0.0
        with amp:
            for i in range(0, D.shape[0], min(16, MICRO)):
                x = D[i:i + min(16, MICRO)].long(); logit = net(x)[:, :-1]; y = x[:, 1:]
                lo += F.cross_entropy(logit.reshape(-1, V).float(), y.reshape(-1), reduction='sum').item()
                t += y.numel()
        return lo / t, t

    MAX = BIAS + COMMON
    state = TrainingState(cfg, net, opt, rng, ckpt)
    try:
        start, history = state.restore(dict(steps=[], vce=[], oce=[]))
        steps, vce, oce = history['steps'], history['vce'], history['oce']
        for step in range(start, MAX):
            for g in opt.param_groups: g['lr'] = lr_at(step)
            s = src(step)
            batch = exposure_step() if s == 'et' else (gstep() if s == 'g' else jstep(step))
            opt.zero_grad(set_to_none=True)
            # One batch draw and one optimizer update: accumulation preserves the paper's batch size.
            for offset in range(0, B, MICRO):
                chunk = batch[offset:offset + MICRO]
                loss = _ce(chunk) * (len(chunk) / B)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}')
                loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            if not torch.isfinite(gn):
                raise FloatingPointError(f'Nonfinite gradient at step {step}')
            opt.step()
            if step % LOG == 0 or step == MAX - 1:
                steps.append(step); vce.append(round(eval_ce(g_te)[0], 4)); oce.append(round(eval_ce(ood_te)[0], 4))
                progress(cfg, step + 1, MAX, target_loss=vce[-1], ood_loss=oce[-1])
            state.after_step(step + 1, dict(steps=steps, vce=vce, oce=oce),
                             force=step + 1 in (BIAS, MAX))
    finally:
        state.close()
    save()                                                 # keep the finished model

    v, nv = eval_ce(g_te); o, no = eval_ce(ood_te)
    final = dict(val_ce=round(v, 4), ood_val_ce=round(o, 4),
                 aggregate=round((v * nv + o * no) / (nv + no), 4),
                 val_bytes=nv, ood_val_bytes=no)
    return dict(arch='foundation', width=W, layers=NL, seq=SEQ, params_M=params_M,
                ord=ORD, seed=SEED, config=cfg['config'], exposure_rate=EXPOSURE_RATE, exposure_set_fraction=EXPOSURE_SET_FRACTION,
                exposure_in_training_pool=True, phase1=BIAS, phase2=COMMON, junk_fraction=CFRAC,
                training_sequences=int(g_tr.shape[0]), exposure_sequences=n_exposure,
                validation=dict(target=good_meta, ood=ood_meta),
                p1_fraction=P1FRAC, good_data=cfg['good_data'], ood_data=cfg['ood_data'],
                good=os.path.splitext(cfg['good_data'])[0], jcorpus=os.path.splitext(cfg['ood_data'])[0],
                optimizer='adam', lr=LR, warmup=WARM, clip=CLIP, batch=B, microbatch=MICRO, checkpoint=ckpt,
                steps=steps, vce=vce, oodce=oce, final=final)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=1088); ap.add_argument('--layers', type=int, default=7)
    ap.add_argument('--seq', type=int, default=512); ap.add_argument('--ord', choices=['goodfirst', 'junkfirst'], default='junkfirst')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--phase1', type=int, default=9000)
    ap.add_argument('--phase2', type=int, default=9000); ap.add_argument('--junk_fraction', type=float, default=0.25)
    ap.add_argument('--junk_volume', type=int, default=2000); ap.add_argument('--exposure_rate', type=float, default=0.0)
    ap.add_argument('--exposure_set_fraction', type=float, default=0.10); ap.add_argument('--p1_fraction', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=1.5e-4); ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--clip', type=float, default=0.5); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--good_data', default='fineweb_345MB.npy'); ap.add_argument('--ood_data', default='code_500MB.npy')
    ap.add_argument('--config', default='100M'); ap.add_argument('--data_dir', default='data')
    ap.add_argument('--log_every', type=int, default=500)
    ap.add_argument('--ckpt_dir', default=None, help='override the automatic per-configuration checkpoint directory')
    ap.add_argument('--microbatch', type=int, default=8)
    ap.add_argument('--checkpoint_every', type=int, default=1000)
    print(json.dumps(run(vars(ap.parse_args()))))
