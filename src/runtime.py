"""Progress reporting and resumable optimizer/RNG training state."""
import hashlib
import json
import os
import random
import signal

import numpy as np
import torch


def training_config(cfg):
    operational = {'ckpt_dir', 'probe_out', 'checkpoint_every', 'resume', 'stop_after',
                   'progress', 'log_every'}
    return {k: v for k, v in cfg.items() if k not in operational}


def config_id(cfg):
    return hashlib.sha256(json.dumps(training_config(cfg), sort_keys=True).encode()).hexdigest()


def progress(cfg, step, total, **metrics):
    if cfg.get('progress', True):
        print(json.dumps(dict(event='progress', run_id=cfg.get('run_id'),
                              step=step, total=total, **metrics)), flush=True)


class TrainingInterrupted(SystemExit):
    """Exit 75 after saving; resubmitting the same command resumes the run."""
    def __init__(self, step):
        self.step = step
        super().__init__(75)


class TrainingState:
    def __init__(self, cfg, net, opt, rng, final_path):
        self.cfg, self.net, self.opt, self.rng = cfg, net, opt, rng
        self.path = final_path + '.resume' if final_path else None
        self.every = int(cfg.get('checkpoint_every', 1000))
        if self.every < 1:
            raise ValueError('checkpoint_every must be positive')
        self.requested_stop = False
        self.handlers = {}
        # Slurm's advance warning and termination signals request a checkpoint
        # after the current optimizer step. No filesystem I/O in the handler.
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            self.handlers[sig] = signal.signal(sig, self._request_stop)

    def _request_stop(self, signum, frame):
        self.requested_stop = True

    def restore(self, default):
        if not self.path or not self.cfg.get('resume', True) or not os.path.exists(self.path):
            return 0, default
        # These are local, self-generated optimizer/RNG checkpoints, not external weights.
        payload = torch.load(self.path, map_location=next(self.net.parameters()).device,
                             weights_only=False)
        if payload.get('config_id') != config_id(self.cfg):
            self.close()
            raise ValueError(f'{self.path}: checkpoint configuration differs; use a new output directory')
        self.net.load_state_dict(payload['model'])
        self.opt.load_state_dict(payload['optimizer'])
        self.rng.bit_generator.state = payload['numpy_generator']
        np.random.set_state(payload['numpy_global'])
        random.setstate(payload['python_random'])
        torch.set_rng_state(payload['torch_rng'].cpu())
        if payload['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in payload['cuda_rng']])
        print(json.dumps(dict(event='resumed', step=payload['step'], checkpoint=self.path)), flush=True)
        return payload['step'], payload['history']

    def save(self, step, history):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        payload = dict(version=1, config_id=config_id(self.cfg), config=training_config(self.cfg),
                       step=step, model=self.net.state_dict(), optimizer=self.opt.state_dict(),
                       numpy_generator=self.rng.bit_generator.state, numpy_global=np.random.get_state(),
                       python_random=random.getstate(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       history=history)
        with open(self.path + '.tmp', 'wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self.path + '.tmp', self.path)

    def after_step(self, step, history, force=False):
        stop_after = self.cfg.get('stop_after')
        stop = self.requested_stop or (stop_after is not None and step >= stop_after)
        if force or step % self.every == 0 or stop:
            self.save(step, history)
        if stop:
            self.close()
            if not self.path:
                raise RuntimeError('Training interrupted without a checkpoint path')
            print(json.dumps(dict(event='checkpointed_stop', step=step, checkpoint=self.path)), flush=True)
            raise TrainingInterrupted(step)

    def close(self):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
        self.handlers.clear()
