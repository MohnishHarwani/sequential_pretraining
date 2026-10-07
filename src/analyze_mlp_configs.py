"""Paired accuracy and cross-phase representational overlap for MLP."""
import argparse
from collections import defaultdict
import contextlib
import csv
import json
from pathlib import Path
import sys
import numpy as np
from mechanism_archive import overlap_scores


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(records, output, make_plots=True, fractions=(.01,.05,.10)):
    fractions=tuple(fractions)
    if not fractions or any(not 0 < f <= 1 for f in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError('Provide distinct overlap fractions in (0, 1]')
    if not records:
        raise ValueError('No MLP records to analyze')
    if any(r['config'] != 'mlp' for r in records):
        raise ValueError('Expected the default MLP experiment')
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    if any(r['arch'] != 'mlp' or r['validation']['validation_count'] != 6222 for r in records):
        raise ValueError('Expected MLP results evaluated on the expanded 6,222-image held-out set')
    for config in sorted({r['config'] for r in records}):
        subset=[r for r in records if r['config']==config]
        out=output/config; out.mkdir(exist_ok=True)
        groups=defaultdict(dict)
        overlaps=[]
        for rec in subset:
            key=(rec['task'], rec['width'], rec['seed'])
            if rec['ord'] in groups[key]:
                raise ValueError(f'Duplicate run {config} {key} {rec["ord"]}')
            groups[key][rec['ord']]=rec
            if rec['ord'] == 'junkfirst':
                for value in overlap_scores(rec['mechanism']['activations'], fractions):
                    overlaps.append(dict(config=config, dataset=rec['task'], width=rec['width'], seed=rec['seed'],
                                         order=rec['ord'], comparison='cross_probe',
                                         fraction=value['fraction'], overlap=value['overlap'],
                                         **{f'layer_{i}':v for i,v in enumerate(value['per_layer'])}))
        paired=[]
        for (dataset,width,seed),arms in sorted(groups.items()):
            if set(arms) != {'goodfirst', 'junkfirst'}:
                raise ValueError(f'Missing ordering for {config}/{dataset}/{width}/{seed}')
            g,j=arms['goodfirst'],arms['junkfirst']
            paired.append(dict(config=config,dataset=dataset,width=width,seed=seed,params_M=g['params_M'],
                               target_first_accuracy=g['val_acc_full'],ood_first_accuracy=j['val_acc_full'],
                               primacy_sensitivity=g['val_acc_full']-j['val_acc_full']))
        summary=[]
        for dataset,width in sorted({(r['dataset'],r['width']) for r in paired}):
            rows=[r for r in paired if (r['dataset'],r['width'])==(dataset,width)]
            if sorted(r['seed'] for r in rows) != [0,1,2]:
                raise ValueError(f'Expected seeds 0/1/2: {config}/{dataset}/{width}')
            row=dict(config=config,dataset=dataset,width=width,params_M=rows[0]['params_M'],seeds=len(rows))
            for metric in ['target_first_accuracy','ood_first_accuracy','primacy_sensitivity']:
                values=[r[metric] for r in rows]
                row[metric+'_mean']=float(np.mean(values))
                row[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
            summary.append(row)
        for name,rows in [('accuracy_by_seed',paired),('accuracy_summary',summary),('overlap_by_seed',overlaps)]:
            write_csv(out/(name+'.csv'),rows)
        if make_plots:
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'figures'))
            from fig_mlp_overlap import plot
            with contextlib.redirect_stdout(sys.stderr):
                plot(out,out/'accuracy_summary.csv',out,config=config,fractions=fractions)
    if make_plots:
        from mlp_plots import from_records, primacy
        primacy(from_records(records, include_overlap=False), output/'mlp_primacy_seeds')
    report=['# MLP\n',f'{len(records)} completed runs.',
            'Training: 56,000 original images. Validation: original 4,000 plus 2,222 official test images.',
            f'{sum("reevaluation" in r for r in records)} records reuse completed weights with a new final evaluation; any reused historical learning curves still use 4,000 images.',
            'Overlap compares Phase-1 OOD activations on 512 corruption-training images (0J) with Phase-2 target activations on 512 held-out target images (TX), for OOD-first runs. Accuracy uses validation.',
            'The overlap figure has threshold panels with means over seeds 0, 1, 2. The primacy figure has three dataset panels and separate seed lines.',
            'Thresholds: '+', '.join(f'{f:.0%}' for f in fractions)+'. All neuron scores and both checkpoints remain available for other thresholds.',
            'Low overlap is observational; neither monotonicity nor a causal neuron-role interpretation is assumed.\n']
    if make_plots:
        for c in sorted({r['config'] for r in records}):
            report.append(f'- [Representational overlap: PNG]({c}/mlp_overlap.png) · [PDF]({c}/mlp_overlap.pdf)')
        report.append('- [Primacy seed curves: PNG](mlp_primacy_seeds.png) · [PDF](mlp_primacy_seeds.pdf)')
    (output/'RESULTS.md').write_text('\n\n'.join(report)+'\n')
    return dict(runs=len(records),configs=sorted({r['config'] for r in records}),validation_count=6222,
                plots=make_plots,fractions=fractions,comparison='cross_probe')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('records',type=Path,help='JSONL run records')
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--fractions',nargs='+',type=float,default=[.01,.05,.10])
    a=ap.parse_args()
    print(json.dumps(analyze([json.loads(line) for line in a.records.read_text().splitlines() if line.strip()],a.output,fractions=a.fractions)))
