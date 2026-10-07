"""Aggregate paired MLP runs and saved overlap measurements into CSVs and figures."""
import argparse, csv, fcntl, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from mechanism_archive import overlap_scores


def write_csv(path, rows):
    with open(str(path)+'.tmp','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(str(path)+'.tmp',path)


def analyze(root):
    root=Path(root)
    with (root/'analysis.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        out=root/'analysis'
        (out/'complete.json').unlink(missing_ok=True)
        manifest=json.loads((root/'manifest.json').read_text())
        records=[json.loads((root/'records'/f"{task['id']}.json").read_text()) for task in manifest['tasks']]
        if any(r.get('arch') != 'mlp' for r in records):
            raise ValueError('MLP overlap analysis requires MLP records')
        if {r.get('config') for r in records} != {'mlp'}:
            raise ValueError('This analyzer expects the default MLP experiment')
        title = 'MLP'
        groups=defaultdict(dict)
        for rec in records:
            groups[(rec['task'],rec['width'],rec['seed'])][rec['ord']]=rec
        paired=[]; overlaps=[]
        for (task,width,seed),arms in sorted(groups.items()):
            g,j=arms['goodfirst'],arms['junkfirst']
            row=dict(dataset=task,width=width,seed=seed,params_M=g['params_M'],
                     target_first_accuracy=g['val_acc_full'],ood_first_accuracy=j['val_acc_full'],
                     primacy_sensitivity=g['val_acc_full']-j['val_acc_full'])
            paired.append(row)
            for ov in overlap_scores(j['mechanism']['activations']):
                overlaps.append(dict(dataset=task,width=width,seed=seed,fraction=ov['fraction'],
                                     overlap=ov['overlap'],**{f'layer_{i}':v for i,v in enumerate(ov['per_layer'])}))
        summary=[]; overlap_summary=[]
        for task,width in sorted({(r['dataset'],r['width']) for r in paired}):
            rows=[r for r in paired if (r['dataset'],r['width'])==(task,width)]
            result=dict(dataset=task,width=width,params_M=rows[0]['params_M'],seeds=len(rows))
            for metric in ['target_first_accuracy','ood_first_accuracy','primacy_sensitivity']:
                values=[r[metric] for r in rows]
                result[metric+'_mean']=float(np.mean(values))
                result[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
            summary.append(result)
            for fraction in [.01,.05,.10]:
                values=[r['overlap'] for r in overlaps if (r['dataset'],r['width'],r['fraction'])==(task,width,fraction)]
                overlap_summary.append(dict(dataset=task,width=width,fraction=fraction,seeds=len(values),
                                            overlap_mean=float(np.mean(values)),
                                            overlap_std=float(np.std(values,ddof=1)) if len(values)>1 else 0.))
        out.mkdir(exist_ok=True)
        for name,rows in [('accuracy_by_seed',paired),('accuracy_summary',summary),
                          ('overlap_by_seed',overlaps),('overlap_summary',overlap_summary)]:
            write_csv(out/(name+'.csv'),rows)
        with (out/'runs.jsonl').open('w') as f:
            for rec in records: f.write(json.dumps(rec)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'mnist':'#0072B2','fashion':'#E69F00','kmnist':'#009E73'}
        labels={'mnist':'MNIST','fashion':'Fashion-MNIST','kmnist':'KMNIST'}
        fig,ax=plt.subplots(figsize=(7,4.5))
        for task,color in colors.items():
            rows=[r for r in summary if r['dataset']==task]
            if not rows: continue
            ax.errorbar([r['params_M']*1e6 for r in rows], [r['primacy_sensitivity_mean'] for r in rows],
                        yerr=[r['primacy_sensitivity_std'] for r in rows],color=color,marker='o',capsize=3,label=labels[task])
        ax.set_xscale('log'); ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel='Parameters',ylabel='Target-first − OOD-first final accuracy',title=title)
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(out/'mlp_sensitivity.png',dpi=200); fig.savefig(out/'mlp_sensitivity.pdf'); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        for ax,fraction in zip(axes,[.01,.05,.10]):
            other=ax.twinx()
            for task,color in colors.items():
                rows=[r for r in summary if r['dataset']==task]
                ovs=[r for r in overlap_summary if r['dataset']==task and r['fraction']==fraction]
                if not rows: continue
                x=[r['params_M']*1e6 for r in rows]
                ax.plot(x,[r['primacy_sensitivity_mean'] for r in rows],'-o',color=color,label=labels[task])
                other.plot(x,[r['overlap_mean'] for r in ovs],'--s',color=color,alpha=.8)
            ax.set_xscale('log'); ax.set(xlabel='Parameters',title=f'Top {fraction:.0%}')
            ax.grid(alpha=.2); other.set_ylim(0,1)
        axes[0].set_ylabel('Primacy sensitivity (solid)')
        other.set_ylabel('Jaccard overlap (dashed)')
        axes[0].legend(); fig.suptitle(title + ': sensitivity and cross-phase overlap')
        fig.tight_layout(); fig.savefig(out/'mlp_overlap.png',dpi=200); fig.savefig(out/'mlp_overlap.pdf'); plt.close(fig)
        metadata=dict(runs=len(records),pairs=len(paired),config='mlp',smoke=manifest.get('smoke',False),
                      error_bars='sample standard deviation across paired seeds',
                      accuracy='final full-held-out accuracy',comparison='cross_probe',
                      overlap='Phase-1 OOD to Phase-2 target (0J/TX), layer mean of top-f Jaccard; OOD-first runs')
        (out/'complete.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return metadata


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('campaign')
    print(json.dumps(analyze(ap.parse_args().campaign),indent=2))
