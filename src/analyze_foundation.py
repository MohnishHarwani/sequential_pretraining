"""Verify foundation campaigns and create Figure-6-style plots and result tables."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict
import numpy as np
from campaign_queue import atomic_json


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(root):
    import torch
    torch.set_num_threads(1)
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['code_sha256'].items():
        assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest() == digest, name
    records = []; groups = defaultdict(list); by_seed = []
    for index, task in enumerate(manifest['tasks'], 1):
        print(f"Verifying {index}/{len(manifest['tasks'])}: {task['id']}", file=sys.stderr, flush=True)
        cfg = task['cfg']; rec = json.loads((root/'records'/f"{task['id']}.json").read_text())
        assert rec['run_id'] == task['id'] and rec['arch'] == 'foundation'
        assert 'overlap' not in rec and 'mechanism' not in rec
        for key in ['config','width','layers','seq','ord','seed','phase1','phase2','exposure_rate',
                    'exposure_set_fraction','p1_fraction','lr','warmup','clip','batch','microbatch',
                    'good_data','ood_data','junk_fraction']:
            assert rec[key] == cfg[key], (task['id'], key)
        assert rec['steps'][-1] == cfg['phase1']+cfg['phase2']-1
        assert rec['exposure_in_training_pool'] is True
        for label, filename in [('target',cfg['good_data']),('ood',cfg['ood_data'])]:
            assert rec['validation'][label] == manifest['data'][filename]
        final = rec['final']
        expected_bytes = math.ceil(50_000/cfg['seq'])*(cfg['seq']-1)
        assert final['val_bytes'] == final['ood_val_bytes'] == expected_bytes
        for key in ['val_ce','ood_val_ce','aggregate']:
            assert math.isfinite(final[key]) and final[key] >= 0
        assert abs(final['aggregate']-(final['val_ce']+final['ood_val_ce'])/2) <= .00011
        payload = torch.load(rec['checkpoint'],map_location='cpu',weights_only=False,mmap=False)
        assert payload['step'] == cfg['phase1']+cfg['phase2']
        assert payload['model']['tok.weight'].shape == (256,cfg['width'])
        for name, value in payload['model'].items():
            if name == 'm': continue  # causal mask intentionally contains -infinity
            assert torch.isfinite(value).all(), (task['id'],name)
        del payload
        good=cfg['good_data'].split('_')[0];ood=cfg['ood_data'].split('_')[0]
        arm='target_first' if cfg['ord']=='goodfirst' else ('ood_first_et' if cfg['exposure_rate'] else 'ood_first')
        row=dict(target=good,ood=ood,arm=arm,seed=cfg['seed'],target_loss=final['val_ce'],
                 ood_loss=final['ood_val_ce'],aggregate_loss=final['aggregate'])
        by_seed.append(row);groups[(good,ood,arm)].append(row);records.append(rec)
    if not manifest['smoke']:
        assert len(records)==72 and len(groups)==24
        assert all(sorted(r['seed'] for r in rows)==[0,1,2] for rows in groups.values())
    summary=[]
    for (good,ood,arm),rows in sorted(groups.items()):
        row=dict(target=good,ood=ood,arm=arm,seeds=len(rows))
        for metric in ['target_loss','ood_loss','aggregate_loss']:
            values=[r[metric] for r in rows]
            row[metric+'_mean']=float(np.mean(values));row[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        summary.append(row)
    out=root/'analysis';out.mkdir(exist_ok=True)
    write_csv(out/'loss_by_seed.csv',by_seed);write_csv(out/'loss_summary.csv',summary)
    with (out/'foundation.jsonl').open('w') as f:
        for rec in records:f.write(json.dumps(rec)+'\n')
    args=[sys.executable,str(root/'code/figures/fig_foundation.py'),'--scale',manifest['scale'],
          '--input',str(out/'foundation.jsonl'),'--out-dir',str(out)]
    if manifest['smoke']:args.append('--smoke')
    subprocess.run(args,check=True,stdout=subprocess.PIPE,text=True)
    lines=[f"# Foundation {manifest['scale']} — Figure 6 replication"+(' — SMOKE ONLY' if manifest['smoke'] else ''),'',
           f"{len(records)} verified runs. Losses are nats per scored byte; lower is better.",
           'Means ± sample standard deviations across seeds. Aggregate loss weights target and OOD bytes equally.','',
           f"[Figure](foundation_{manifest['scale']}.png) · [PDF](foundation_{manifest['scale']}.pdf)",'',
           '| Target | OOD | Arm | Seeds | Target loss | Aggregate loss |',
           '| --- | --- | --- | ---: | ---: | ---: |']
    for r in summary:
        lines.append(f"| {r['target']} | {r['ood']} | {r['arm']} | {r['seeds']} | "
                     f"{r['target_loss_mean']:.4f} ± {r['target_loss_std']:.4f} | "
                     f"{r['aggregate_loss_mean']:.4f} ± {r['aggregate_loss_std']:.4f} |")
    lines += ['', '[Per-seed CSV](loss_by_seed.csv) · [Summary CSV](loss_summary.csv)',
              'Final model weights and optimizer/RNG resume checkpoints are retained in `../runs/`.',
              'Pinned dataset revisions, source-shard order, checksums and shared held-out split are in `../manifest.json`.',
              'All models are byte-level, with a single shared output head. No overlap mechanism is calculated.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    result=dict(verified_at=time.time(),runs=len(records),groups=len(groups),smoke=manifest['smoke'],
                source_hashes='passed',final_weights='passed',phase_steps='passed',
                shared_validation='passed',seed_coverage='passed',figure=str(out/f"foundation_{manifest['scale']}.png"))
    atomic_json(out/'verification.json',result);atomic_json(out/'complete.json',result)
    return result


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('campaign')
    print(json.dumps(analyze(ap.parse_args().campaign),indent=2))
